from __future__ import annotations

from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_runtime_v2 as runtime,
)


def _jsonl(rows):
    return b"".join(runtime.v1.integration_v1.canonical_json(row) + b"\n" for row in rows)


def _fixture_pack(base: Path):
    root = base / runtime.v1.TRAIN_SOURCE_ROOT_RELATIVE
    root.mkdir(parents=True)
    view_rows = []
    label_rows = []
    for ordinal in range(runtime.v1.ITEM_COUNT):
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
    view_path = root / "train_integration.view.jsonl"
    label_path = root / "train_integration.labels.jsonl"
    view_path.write_bytes(_jsonl(view_rows))
    label_path.write_bytes(_jsonl(label_rows))
    integration = {
        "TRAIN_diagnostic_pack": {
            "item_count": runtime.v1.ITEM_COUNT,
            "label_file_sha256": runtime.v1.integration_v1.file_sha256(label_path),
            "label_file_size_bytes": label_path.stat().st_size,
            "view_file_sha256": runtime.v1.integration_v1.file_sha256(view_path),
            "view_file_size_bytes": view_path.stat().st_size,
        }
    }
    return integration, label_path


def test_load_views_does_not_open_label_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integration, label_path = _fixture_pack(tmp_path)
    original = Path.read_bytes
    opened = []

    def tracking(self):
        opened.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", tracking)
    items = runtime.load_train_views(tmp_path, integration)
    assert len(items) == runtime.v1.ITEM_COUNT
    assert label_path not in opened


def test_labels_refuse_to_open_without_action_seal(tmp_path: Path) -> None:
    integration, _ = _fixture_pack(tmp_path)
    items = runtime.load_train_views(tmp_path, integration)
    with pytest.raises(runtime.FiqaTrainRuntimeV2Error):
        runtime.load_train_labels_after_seal(
            base=tmp_path,
            integration=integration,
            items=items,
            action_path=tmp_path / "absent.actions.json",
            expected_action_file_sha256="0" * 64,
        )


def test_labels_open_only_after_bound_action_seal(tmp_path: Path) -> None:
    integration, _ = _fixture_pack(tmp_path)
    items = runtime.load_train_views(tmp_path, integration)
    action_path = tmp_path / "actions.json"
    action_path.write_bytes(b"sealed\n")
    digest = runtime.v1.integration_v1.file_sha256(action_path)
    labels = runtime.load_train_labels_after_seal(
        base=tmp_path,
        integration=integration,
        items=items,
        action_path=action_path,
        expected_action_file_sha256=digest,
    )
    assert len(labels) == runtime.v1.ITEM_COUNT
    assert labels[items[0].item_key] == ("d0",)


def test_label_loader_rejects_action_drift(tmp_path: Path) -> None:
    integration, _ = _fixture_pack(tmp_path)
    items = runtime.load_train_views(tmp_path, integration)
    action_path = tmp_path / "actions.json"
    action_path.write_bytes(b"sealed\n")
    with pytest.raises(runtime.FiqaTrainRuntimeV2Error):
        runtime.load_train_labels_after_seal(
            base=tmp_path,
            integration=integration,
            items=items,
            action_path=action_path,
            expected_action_file_sha256="f" * 64,
        )


def test_bound_artifact_verifier_checks_all_nine_files(tmp_path: Path) -> None:
    rows = []
    for index in range(9):
        path = tmp_path / f"artifact-{index}.bin"
        path.write_bytes(str(index).encode())
        rows.append(
            {
                "relative_path": path.name,
                "sha256": runtime.v1.integration_v1.file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    failure = {"label_free_artifact_bindings": rows}
    observed = runtime._verify_label_free_artifacts(tmp_path, failure)
    assert len(observed) == 9


def test_bound_artifact_verifier_rejects_byte_drift(tmp_path: Path) -> None:
    rows = []
    for index in range(9):
        path = tmp_path / f"artifact-{index}.bin"
        path.write_bytes(str(index).encode())
        rows.append(
            {
                "relative_path": path.name,
                "sha256": runtime.v1.integration_v1.file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    (tmp_path / "artifact-3.bin").write_bytes(b"changed")
    with pytest.raises(runtime.FiqaTrainRuntimeV2Error):
        runtime._verify_label_free_artifacts(
            tmp_path,
            {"label_free_artifact_bindings": rows},
        )


def test_bound_artifact_verifier_applies_exact_impossible_hash_correction(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(9):
        path = tmp_path / f"artifact-{index}.bin"
        path.write_bytes(str(index).encode())
        digest = runtime.v1.integration_v1.file_sha256(path)
        rows.append(
            {
                "relative_path": path.name,
                "sha256": digest + ("f" if index == 3 else ""),
                "size_bytes": path.stat().st_size,
            }
        )
    corrected = rows[3]["sha256"][:64]
    observed = runtime._verify_label_free_artifacts(
        tmp_path,
        {"label_free_artifact_bindings": rows},
        {
            "correction": {
                "correct_sha256": corrected,
                "erroneous_declared_sha256": rows[3]["sha256"],
                "relative_path": rows[3]["relative_path"],
            }
        },
    )
    assert len(observed) == 9


def test_bound_artifact_verifier_rejects_unregistered_impossible_hash(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(9):
        path = tmp_path / f"artifact-{index}.bin"
        path.write_bytes(str(index).encode())
        digest = runtime.v1.integration_v1.file_sha256(path)
        rows.append(
            {
                "relative_path": path.name,
                "sha256": digest + ("f" if index == 3 else ""),
                "size_bytes": path.stat().st_size,
            }
        )
    with pytest.raises(runtime.FiqaTrainRuntimeV2Error):
        runtime._verify_label_free_artifacts(
            tmp_path,
            {"label_free_artifact_bindings": rows},
        )


def test_hipporag_replay_requires_all_inputs(tmp_path: Path) -> None:
    previous_root = tmp_path / "previous"
    previous_root.mkdir()
    new_roots = []
    for ordinal in range(runtime.v1.ITEM_COUNT - 1):
        previous_item = previous_root / f"item_{ordinal:03d}"
        new_item = tmp_path / "new" / f"item_{ordinal:03d}"
        previous_item.mkdir(parents=True)
        new_item.mkdir(parents=True)
        (previous_item / "input.json").write_bytes(b"{}\n")
        (new_item / "input.json").write_bytes(b"{}\n")
        new_roots.append(new_item)
    with pytest.raises(runtime.FiqaTrainRuntimeV2Error):
        runtime._verify_replayed_hipporag_inputs(
            previous_root=previous_root,
            new_roots=new_roots,
        )
