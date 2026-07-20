from __future__ import annotations

from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_dev_runtime_v1 as runtime,
)


def _jsonl(rows):
    return b"".join(
        runtime.train_v1.integration_v1.canonical_json(row) + b"\n" for row in rows
    )


def _fixture_pack(base: Path):
    root = base / runtime.ACQUISITION_ROOT_RELATIVE
    root.mkdir(parents=True)
    view_rows = []
    label_rows = []
    for ordinal in range(runtime.ITEM_COUNT):
        key = f"{ordinal:064x}"
        view_rows.append(
            {
                "excluded_document_ids": [],
                "family": "FIQA",
                "item_key": key,
                "query": f"query {ordinal}",
                "source_query_id": f"q{ordinal}",
            }
        )
        label_rows.append(
            {
                "family": "FIQA",
                "gold_document_ids": [f"d{ordinal}"],
                "item_key": key,
            }
        )
    view_path = root / "C_confirm.view.jsonl"
    label_path = root / "C_confirm.labels.jsonl"
    view_path.write_bytes(_jsonl(view_rows))
    label_path.write_bytes(_jsonl(label_rows))
    acquisition_result = {
        "C_confirm_pack": {
            "item_count": runtime.ITEM_COUNT,
            "label_file_sha256": runtime.train_v1.integration_v1.file_sha256(
                label_path
            ),
            "label_file_size_bytes": label_path.stat().st_size,
            "view_file_sha256": runtime.train_v1.integration_v1.file_sha256(
                view_path
            ),
            "view_file_size_bytes": view_path.stat().st_size,
        }
    }
    return acquisition_result, label_path


def test_load_views_does_not_open_dev_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition_result, label_path = _fixture_pack(tmp_path)
    original = Path.read_bytes
    opened = []

    def tracking(self):
        opened.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", tracking)
    items = runtime.load_dev_views(tmp_path, acquisition_result)
    assert len(items) == runtime.ITEM_COUNT
    assert label_path not in opened


def test_labels_refuse_without_bound_action_seal(tmp_path: Path) -> None:
    acquisition_result, _ = _fixture_pack(tmp_path)
    items = runtime.load_dev_views(tmp_path, acquisition_result)
    with pytest.raises(runtime.FiqaDevRuntimeError):
        runtime.load_dev_labels_after_seal(
            base=tmp_path,
            acquisition_result=acquisition_result,
            items=items,
            action_path=tmp_path / "absent.actions.json",
            expected_action_file_sha256="0" * 64,
        )


def test_labels_open_only_after_exact_action_seal(tmp_path: Path) -> None:
    acquisition_result, _ = _fixture_pack(tmp_path)
    items = runtime.load_dev_views(tmp_path, acquisition_result)
    action_path = tmp_path / "actions.json"
    action_path.write_bytes(b"sealed\n")
    digest = runtime.train_v1.integration_v1.file_sha256(action_path)
    labels = runtime.load_dev_labels_after_seal(
        base=tmp_path,
        acquisition_result=acquisition_result,
        items=items,
        action_path=action_path,
        expected_action_file_sha256=digest,
    )
    assert len(labels) == runtime.ITEM_COUNT
    assert labels[items[0].item_key] == ("d0",)


def test_primary_decision_requires_both_strict_positive_deltas() -> None:
    p10 = [3] * runtime.ITEM_COUNT
    raw = [2] * runtime.ITEM_COUNT
    hippo = [1] * runtime.ITEM_COUNT
    passed, paired = runtime.primary_decision(
        {"P10": p10, "RAW": raw, "HippoRAG": hippo}
    )
    assert passed is True
    assert paired["P10_minus_RAW"]["net_integer_ndcg"] > 0
    failed, tied = runtime.primary_decision(
        {"P10": p10, "RAW": p10, "HippoRAG": hippo}
    )
    assert failed is False
    assert tied["P10_minus_RAW"]["net_integer_ndcg"] == 0


def test_primary_decision_rejects_wrong_item_count() -> None:
    with pytest.raises(runtime.FiqaDevRuntimeError):
        runtime.primary_decision(
            {"P10": [1], "RAW": [0], "HippoRAG": [0]}
        )
