from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from replication_runtime.qasc_nli_v1.binding import (
    ASSET_GIT_COMMIT,
    ASSET_SELF_SHA256,
    FORMAL_WORKER_COUNT,
    NLIWorkerPool,
    score_pairs_in_subprocess,
    verify_design_binding,
    verify_runtime_asset,
    verify_runtime_binding,
)
from replication_runtime.qasc_nli_v1.contract import (
    QASCNLIError,
    canonical_json_line,
    decode_request,
    decode_response,
    encode_request,
)


PROJECT = Path(__file__).parents[1]
ASSET = PROJECT / "manifests/qasc_nli_runtime_asset_v1.json"
MODEL = PROJECT / "artifacts/qasc_nli_runtime_v3/model"


def _require_local_asset() -> None:
    if not MODEL.is_dir():
        pytest.skip("ignored pinned QASC NLI asset is not materialized")


def test_exact_label_free_wire_contract_rejects_qa_fields_and_noncanonical_json() -> None:
    pairs = [{"premise": "Minerals are matter.", "hypothesis": "Quartz is matter."}]
    raw = encode_request(pairs)
    assert decode_request(raw)[0].premise == "Minerals are matter."
    contaminated = {
        "schema": "qasc_nli_pair_score_request_v1",
        "pairs": [
            {
                "premise": "Minerals are matter.",
                "hypothesis": "Quartz is matter.",
                "answerKey": "A",
            }
        ],
    }
    with pytest.raises(QASCNLIError, match="only premise and hypothesis"):
        decode_request(canonical_json_line(contaminated))
    with pytest.raises(QASCNLIError, match="not canonical"):
        decode_request(json.dumps(json.loads(raw)).encode("ascii") + b"\n")


def test_response_parser_fails_closed_on_shape_type_nonfinite_and_extra_fields() -> None:
    valid = canonical_json_line(
        {"schema": "qasc_nli_integer_margin_response_v1", "scores": [3, -4]}
    )
    assert decode_response(valid, expected_count=2) == (3, -4)
    for payload in (
        {"schema": "qasc_nli_integer_margin_response_v1", "scores": [3.0]},
        {"schema": "qasc_nli_integer_margin_response_v1", "scores": [float("nan")]},
        {
            "schema": "qasc_nli_integer_margin_response_v1",
            "scores": [3],
            "answer": "A",
        },
    ):
        with pytest.raises(QASCNLIError):
            decode_response(canonical_json_line(payload), expected_count=1)


def test_committed_asset_and_live_model_tree_verify_exactly() -> None:
    _require_local_asset()
    receipt = verify_runtime_binding(asset_manifest_path=ASSET, model_root=MODEL)
    assert receipt["asset_sha256"] == ASSET_SELF_SHA256
    assert receipt["asset_git_commit"] == ASSET_GIT_COMMIT
    assert receipt["status"] == "verified_offline_immutable_runtime"


def test_asset_manifest_and_model_root_symlink_fail_closed(tmp_path: Path) -> None:
    _require_local_asset()
    manifest_link = tmp_path / "asset.json"
    manifest_link.symlink_to(ASSET)
    with pytest.raises(QASCNLIError, match="symlink"):
        verify_runtime_binding(asset_manifest_path=manifest_link, model_root=MODEL)
    model_link = tmp_path / "model"
    model_link.symlink_to(MODEL, target_is_directory=True)
    with pytest.raises(QASCNLIError, match="symlink"):
        verify_runtime_binding(asset_manifest_path=ASSET, model_root=model_link)


def test_asset_semantic_or_file_drift_fails_before_model_load(tmp_path: Path) -> None:
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    payload["execution"]["batch_size"] = 63
    drift = tmp_path / "drift.json"
    drift.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(QASCNLIError, match="manifest file drifted"):
        verify_runtime_binding(asset_manifest_path=drift, model_root=MODEL)


def test_design_freezes_exact_eight_by_four_worker_profile() -> None:
    design = verify_design_binding(PROJECT)
    assert design["formal_NLI_workers"] == 8
    assert design["torch_threads_per_worker"] == 4
    receipt = verify_runtime_asset(PROJECT, MODEL) if MODEL.is_dir() else None
    if receipt is not None:
        assert receipt["asset_sha256"] == ASSET_SELF_SHA256
    signature = inspect.signature(NLIWorkerPool)
    assert signature.parameters["workers"].default == FORMAL_WORKER_COUNT
    with pytest.raises(QASCNLIError, match="exactly 8"):
        NLIWorkerPool(
            MODEL,
            workers=True,
            project_root=PROJECT,
        )


def test_score_items_keeps_keys_local_and_preserves_exact_order() -> None:
    pool = object.__new__(NLIWorkerPool)
    pool.score_batches = lambda batches: tuple(  # type: ignore[method-assign]
        tuple(range(len(batch))) for batch in batches
    )
    result = pool.score_items(
        [
            ("item-b", [("premise b", "hypothesis b")]),
            (
                "item-a",
                [
                    ("premise a1", "hypothesis a1"),
                    ("premise a2", "hypothesis a2"),
                ],
            ),
        ]
    )
    assert list(result) == ["item-b", "item-a"]
    assert result == {"item-b": (0,), "item-a": (0, 1)}


def test_independent_worker_reproduces_canary_before_scoring_exact_pairs(tmp_path: Path) -> None:
    _require_local_asset()
    pairs = [
        {
            "premise": "Quartz is a mineral. Every mineral is matter.",
            "hypothesis": "Quartz is matter.",
        },
        {
            "premise": "Quartz is a mineral. Every mineral is matter.",
            "hypothesis": "Quartz is not matter.",
        },
    ]
    scores = score_pairs_in_subprocess(
        pairs,
        asset_manifest_path=ASSET,
        model_root=MODEL,
        temporary_root=tmp_path,
    )
    assert len(scores) == 2
    assert all(isinstance(score, int) and not isinstance(score, bool) for score in scores)
    assert scores[0] > scores[1]
