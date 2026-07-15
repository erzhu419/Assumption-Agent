from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from assumption_agent.benchmarks.semantic_assignment_operator_v1 import (
    ALL_DESTINATIONS,
    EMBEDDING_DIMENSION,
    FIT_CONFIGURATION,
    PUBLIC_DEFAULT_DESTINATION,
    SEMANTIC_ASSIGNMENT_ASSET_VERSION,
    SEMANTIC_ASSIGNMENT_OPERATOR_VERSION,
    TARGET_DESTINATIONS,
    SemanticAssignmentError,
    build_semantic_assignment_plan,
    load_operator_asset,
)
from assumption_agent.models import stable_hash


def _operator_asset() -> dict[str, object]:
    coefficients = np.zeros((4, EMBEDDING_DIMENSION), dtype=np.float64)
    for index in range(4):
        coefficients[index, index] = 1.0
    intercepts = np.asarray([-0.5, -0.5, -0.5, -0.5], dtype=np.float64)
    parameter_hash = hashlib.sha256(
        coefficients.astype("<f8").tobytes(order="C")
        + intercepts.astype("<f8").tobytes(order="C")
    ).hexdigest()
    asset: dict[str, object] = {
        "asset_version": SEMANTIC_ASSIGNMENT_ASSET_VERSION,
        "operator_version": SEMANTIC_ASSIGNMENT_OPERATOR_VERSION,
        "target_destinations": list(TARGET_DESTINATIONS),
        "public_default_destination": PUBLIC_DEFAULT_DESTINATION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "fit_configuration": dict(FIT_CONFIGURATION),
        "coefficients": coefficients.tolist(),
        "intercepts": intercepts.tolist(),
        "parameter_bytes_sha256": parameter_hash,
        "candidate_id": stable_hash(
            {
                "operator_version": SEMANTIC_ASSIGNMENT_OPERATOR_VERSION,
                "parameter_bytes_sha256": parameter_hash,
                "train_pack_manifest_hash": "a" * 64,
                "runtime_asset_manifest_hash": "d" * 64,
                "fit_configuration": dict(FIT_CONFIGURATION),
            }
        ),
        "train_pack_manifest_hash": "a" * 64,
        "train_records_hash": "b" * 64,
        "runtime_asset_manifest_hash": "d" * 64,
        "runtime_required_file_set_hash": "e" * 64,
        "fit_source_object_set_hash": "f" * 64,
        "fit_record_count": 5,
        "fit_iterations": [1, 1, 1, 1],
        "consumed_train_resubstitution_correct": 5,
        "consumed_train_resubstitution_total": 5,
        "prospective_claim_authorized": False,
        "raw_extracted_text_persisted": False,
    }
    asset["manifest_hash"] = stable_hash(asset)
    return asset


def _evidence_payload() -> dict[str, object]:
    files = []
    for index, text in enumerate(("llm", "unknown")):
        raw = text.encode()
        files.append(
            {
                "file_id": hashlib.sha256(f"file-{index}".encode()).hexdigest(),
                "filename": f"paper-{index}.pdf",
                "content_sha256": hashlib.sha256(
                    f"content-{index}".encode()
                ).hexdigest(),
                "size_bytes": 10,
                "media_type": "pdf",
                "extraction_status": "ok",
                "evidence": [
                    {
                        "evidence_id": hashlib.sha256(
                            f"evidence-{index}".encode()
                        ).hexdigest(),
                        "kind": "pdf_first_pages_text",
                        "text": text,
                        "text_sha256": hashlib.sha256(raw).hexdigest(),
                        "truncated": False,
                    }
                ],
            }
        )
    body: dict[str, object] = {
        "runtime_policy": "typed_assignment_prepare_plan_apply_reconcile_v3",
        "contract_hash": "1" * 64,
        "destinations": list(ALL_DESTINATIONS),
        "public_default": PUBLIC_DEFAULT_DESTINATION,
        "extraction_policy": {"pages": 2},
        "files": files,
    }
    body["evidence_set_hash"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return body


def test_closed_target_scores_use_public_default_without_confidence_gate() -> None:
    evidence = _evidence_payload()
    asset = _operator_asset()

    def encoder(texts: list[str]) -> np.ndarray:
        assert texts == ["llm", "unknown"]
        values = np.zeros((2, EMBEDDING_DIMENSION), dtype=np.float32)
        values[0, 0] = 1.0
        values[1, 4] = 1.0
        return values

    plan, receipt = build_semantic_assignment_plan(
        evidence_payload=evidence,
        operator_asset=asset,
        encoder=encoder,
    )

    assert [row["destination"] for row in plan["assignments"]] == [
        "LLM",
        "music_history",
    ]
    assert all(
        row["basis"] == "positive_content_evidence"
        for row in plan["assignments"]
    )
    assert receipt["destination_distribution"]["LLM"] == 1
    assert receipt["destination_distribution"]["music_history"] == 1
    serialized = json.dumps(receipt, sort_keys=True)
    assert "unknown" not in serialized
    assert "paper-0" not in serialized
    assert receipt["operator_created_extracted_text_artifact"] is False
    assert receipt["online_calls"] == 0


def test_unavailable_extraction_uses_public_default_basis() -> None:
    evidence = _evidence_payload()
    row = evidence["files"][1]
    row["extraction_status"] = "unavailable"
    row["evidence"] = []
    body = dict(evidence)
    del body["evidence_set_hash"]
    evidence["evidence_set_hash"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    def encoder(texts: list[str]) -> np.ndarray:
        values = np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)
        values[0, 0] = 1.0
        return values

    plan, receipt = build_semantic_assignment_plan(
        evidence_payload=evidence,
        operator_asset=_operator_asset(),
        encoder=encoder,
    )
    assert plan["assignments"][1] == {
        "file_id": row["file_id"],
        "destination": PUBLIC_DEFAULT_DESTINATION,
        "basis": "public_default",
        "evidence_ids": [],
    }
    assert receipt["public_default_unavailable_count"] == 1


def test_evidence_hash_and_text_hash_tampering_fail_closed() -> None:
    evidence = _evidence_payload()
    evidence["files"][0]["evidence"][0]["text"] = "tampered"
    with pytest.raises(SemanticAssignmentError, match="evidence set hash mismatch"):
        build_semantic_assignment_plan(
            evidence_payload=evidence,
            operator_asset=_operator_asset(),
            encoder=lambda _: np.empty((0, EMBEDDING_DIMENSION)),
        )


def test_operator_asset_loader_rejects_parameter_tamper(tmp_path: Path) -> None:
    asset = _operator_asset()
    path = tmp_path / "asset.json"
    path.write_text(json.dumps(asset), encoding="utf-8")
    assert load_operator_asset(path)["candidate_id"] == asset["candidate_id"]

    tampered = copy.deepcopy(asset)
    tampered["coefficients"][0][0] = 2.0
    tampered["manifest_hash"] = stable_hash(
        {key: value for key, value in tampered.items() if key != "manifest_hash"}
    )
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(SemanticAssignmentError, match="parameter hash mismatch"):
        load_operator_asset(path)
