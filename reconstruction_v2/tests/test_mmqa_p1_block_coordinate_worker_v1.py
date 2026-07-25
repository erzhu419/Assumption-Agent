from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import stat

import numpy as np
import pytest

from assumption_agent.benchmarks import mmqa_p1_action_integration_v1 as integration
from assumption_agent.benchmarks import mmqa_p1_block_coordinate_worker_v1 as worker
from assumption_agent.benchmarks import mmqa_p1_local_action_executor_v1 as executor


def _work_id(index: int) -> str:
    return f"mmqa-work-v1-{index:064x}"


def _anonymous_item(index: int, *, pairs: int = 20) -> integration.AnonymousWorkItem:
    return integration.validate_anonymous_work_item(
        {
            "schema": integration.ANONYMOUS_WORK_ITEM_SCHEMA,
            "question": f"Which Aurora {index} launch year is established?",
            "rows": [
                {
                    "ordinal": ordinal,
                    "serialized_content": (
                        f"Aurora {index} row {ordinal} | year {2000 + ordinal}"
                    ),
                }
                for ordinal in range(pairs)
            ],
            "texts": [
                {
                    "ordinal": 100 + ordinal,
                    "serialized_content": (
                        f"Aurora {index} evidence {ordinal} was recorded "
                        f"in {2000 + ordinal}."
                    ),
                }
                for ordinal in range(pairs)
            ],
            "exact_row_text_links": [
                {"row_ordinal": ordinal, "text_ordinal": 100 + ordinal}
                for ordinal in range(pairs)
            ],
        }
    )


def _block(*, pairs: int = 20) -> tuple[worker.AnonymousBlockItem, ...]:
    return tuple(
        worker.AnonymousBlockItem(_work_id(index), _anonymous_item(index, pairs=pairs))
        for index in (1, 2)
    )


def _binding(role: str, *, runtime: str = "a" * 64) -> worker.BlockModelBinding:
    return worker.BlockModelBinding(
        role=role,
        model_path=(
            "/frozen/models/all-MiniLM-L6-v2"
            if role == worker.ROLE_MINILM
            else "/frozen/models/ms-marco-MiniLM-L-6-v2"
        ),
        required_tree_sha256=worker.ROLE_REQUIRED_TREE_SHA256[role],
        local_runtime_identity_sha256=runtime,
    )


class _FakeInitializer:
    def __init__(self) -> None:
        self.initializations: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object):
        self.initializations.append(kwargs)
        role = kwargs["role"]

        def backend(**batch: object) -> object:
            self.calls.append(batch)
            if role == worker.ROLE_MINILM:
                texts = batch["texts"]
                vectors = []
                for index, text in enumerate(texts):  # type: ignore[union-attr]
                    value = 0.05 + (len(text) % 17) / 20.0  # type: ignore[arg-type]
                    vector = [1.0, value + index / 1000.0]
                    vector.extend([0.0] * (executor.MINILM_EMBEDDING_DIMENSION - 2))
                    vectors.append(vector)
                return np.asarray(vectors, dtype=np.float32)
            pairs = batch["pairs"]
            return np.asarray(
                [
                    (len(question) - len(document)) / 100.0
                    for question, document in pairs  # type: ignore[union-attr]
                ],
                dtype=np.float32,
            )

        return backend


def _run_both(
    items: tuple[worker.AnonymousBlockItem, ...] | None = None,
) -> tuple[
    worker.BlockCoordinateArchive,
    worker.BlockCoordinateArchive,
    _FakeInitializer,
    _FakeInitializer,
]:
    items = items or _block()
    mini_fake = _FakeInitializer()
    ce_fake = _FakeInitializer()
    mini = worker.run_block_coordinate_worker(
        items,
        model_binding=_binding(worker.ROLE_MINILM),
        initialize_model=mini_fake,
    )
    ce = worker.run_block_coordinate_worker(
        items,
        model_binding=_binding(worker.ROLE_CROSS_ENCODER),
        initialize_model=ce_fake,
    )
    return mini, ce, mini_fake, ce_fake


def _rehash(payload: dict[str, object], hash_field: str) -> dict[str, object]:
    body = {key: value for key, value in payload.items() if key != hash_field}
    payload[hash_field] = worker._semantic_hash(body)  # noqa: SLF001
    return payload


def _write_canonical(path: Path, payload: object) -> None:
    path.write_bytes(worker._canonical_json_bytes(payload, newline=True))  # noqa: SLF001
    path.chmod(0o600)


def _all_keys(value: object) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        output.update(str(key) for key in value)
        for nested in value.values():
            output.update(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            output.update(_all_keys(nested))
    return output


def test_block_payload_is_canonical_opaque_and_strictly_round_trips(
    tmp_path: Path,
) -> None:
    items = _block(pairs=3)
    payload = worker.anonymous_block_payload(items)
    assert payload["item_count"] == 2
    assert payload["block_sha256"] == worker._semantic_hash(  # noqa: SLF001
        {key: value for key, value in payload.items() if key != "block_sha256"}
    )
    assert worker.validate_anonymous_block_payload(payload) == items

    path = tmp_path / "anonymous-block.private.json"
    _write_canonical(path, payload)
    assert worker.load_anonymous_block(path) == items
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    reordered = copy.deepcopy(payload)
    reordered["items"] = list(reversed(reordered["items"]))  # type: ignore[arg-type]
    with pytest.raises(worker.MmqaP1BlockCoordinateWorkerError, match="identity"):
        worker.validate_anonymous_block_payload(reordered)


def test_private_anonymous_block_writer_is_exclusive_canonical_0600(
    tmp_path: Path,
) -> None:
    items = _block(pairs=3)
    path = tmp_path / "shared-two-cli-input.private.json"
    file_sha256 = worker.write_private_anonymous_block(path, items)
    expected = worker._canonical_json_bytes(  # noqa: SLF001
        worker.anonymous_block_payload(items), newline=True
    )
    assert path.read_bytes() == expected
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert file_sha256 == hashlib.sha256(expected).hexdigest()
    assert worker.load_anonymous_block(path) == items
    with pytest.raises(worker.MmqaP1BlockCoordinateWorkerError, match="exists"):
        worker.write_private_anonymous_block(path, items)


def test_block_rejects_duplicate_or_nonopaque_work_ids() -> None:
    item = _anonymous_item(1, pairs=3)
    duplicate = (
        worker.AnonymousBlockItem(_work_id(1), item),
        worker.AnonymousBlockItem(_work_id(1), item),
    )
    with pytest.raises(worker.MmqaP1BlockCoordinateWorkerError, match="duplicate"):
        worker.validate_block_items(duplicate)
    with pytest.raises(worker.MmqaP1BlockCoordinateWorkerError, match="opaque"):
        worker.AnonymousBlockItem("source-qid-123", item)


def test_minilm_initializes_once_and_uses_only_fixed_bounded_chunks() -> None:
    items = _block()
    fake = _FakeInitializer()
    archive = worker.run_block_coordinate_worker(
        items,
        model_binding=_binding(worker.ROLE_MINILM),
        initialize_model=fake,
    )
    # 2 * (one question + 40 units) = 82 inputs, frozen batch 32.
    assert len(fake.initializations) == 1
    assert [len(call["texts"]) for call in fake.calls] == [32, 32, 18]
    assert all(call["batch_size"] == 32 for call in fake.calls)
    assert all(call["max_length"] == 256 for call in fake.calls)
    assert all(call["device"] == "cuda:0" for call in fake.calls)
    assert all(call["deterministic"] is True for call in fake.calls)
    assert fake.initializations[0]["local_files_only"] is True
    assert fake.initializations[0]["trust_remote_code"] is False
    assert fake.initializations[0]["network_disabled"] is True
    assert archive.model_initialization_count == 1
    assert archive.inference_input_count == 82
    assert archive.unit_count == 80
    assert archive.batch_call_count == 3
    assert archive.frozen_batch_size == 32
    assert tuple((row.work_id, row.ordinal) for row in archive.rows) == (
        worker._expected_row_keys(items)  # noqa: SLF001
    )
    assert all(0.0 <= row.coordinate <= 1.0 for row in archive.rows)


def test_cross_encoder_initializes_once_and_uses_fixed_bounded_chunks() -> None:
    items = _block()
    fake = _FakeInitializer()
    archive = worker.run_block_coordinate_worker(
        items,
        model_binding=_binding(worker.ROLE_CROSS_ENCODER),
        initialize_model=fake,
    )
    # 80 unit pairs, frozen batch 64.
    assert len(fake.initializations) == 1
    assert [len(call["pairs"]) for call in fake.calls] == [64, 16]
    assert all(call["batch_size"] == 64 for call in fake.calls)
    assert all(call["max_length"] == 512 for call in fake.calls)
    assert all(call["device"] == "cuda:1" for call in fake.calls)
    assert archive.model_initialization_count == 1
    assert archive.inference_input_count == 80
    assert archive.unit_count == 80
    assert archive.batch_call_count == 2
    expected_first = 1.0 / (
        1.0
        + math.exp(
            -(
                len(items[0].work_item.question)
                - len(items[0].work_item.units[0].serialized_content)
            )
            / 100.0
        )
    )
    assert archive.rows[0].coordinate == pytest.approx(expected_first)


def test_two_role_archives_are_text_free_private_coordinate_only() -> None:
    mini, ce, _mini_fake, _ce_fake = _run_both()
    forbidden = {
        "question",
        "text",
        "content",
        "source",
        "gold",
        "answer",
        "support",
        "family",
        "qid",
    }
    for archive in (mini, ce):
        payload = archive.payload()
        assert not (_all_keys(payload) & forbidden)
        rendered = json.dumps(payload, sort_keys=True)
        assert "Which Aurora" not in rendered
        assert "evidence" not in rendered
        assert "/frozen/models" not in rendered
        assert payload["dynamic_batch_resize_count"] == 0
        assert payload["retry_replay_resample_count"] == 0
        assert payload["network_or_api_call_count"] == 0
        assert payload["model_initialization_count"] == 1


def test_private_archive_is_exclusive_canonical_mode_0600(
    tmp_path: Path,
) -> None:
    mini, _ce, _mini_fake, _ce_fake = _run_both(_block(pairs=3))
    path = tmp_path / "minilm.private.json"
    file_sha = worker.write_private_coordinate_archive(path, mini)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_bytes() == worker._canonical_json_bytes(  # noqa: SLF001
        mini.payload(), newline=True
    )
    assert len(file_sha) == 64
    loaded = worker.load_coordinate_archive(path)
    assert loaded.payload() == mini.payload()
    with pytest.raises(worker.MmqaP1BlockCoordinateWorkerError, match="exists"):
        worker.write_private_coordinate_archive(path, mini)


@pytest.mark.parametrize("mutation", ["reordered", "missing", "duplicate", "added"])
def test_controller_rejects_reordered_missing_duplicate_or_added_rows(
    mutation: str,
) -> None:
    items = _block(pairs=3)
    mini, _ce, _mini_fake, _ce_fake = _run_both(items)
    payload = copy.deepcopy(mini.payload())
    rows = payload["rows"]
    assert isinstance(rows, list)
    if mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "missing":
        rows.pop()
        payload["unit_count"] = int(payload["unit_count"]) - 1
    elif mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    else:
        extra = copy.deepcopy(rows[-1])
        extra["ordinal"] = 999
        rows.append(extra)
        payload["unit_count"] = int(payload["unit_count"]) + 1
    _rehash(payload, "archive_sha256")
    archive = worker.parse_coordinate_archive_payload(payload)
    with pytest.raises(
        worker.MmqaP1BlockCoordinateWorkerError,
        match="missing, duplicated, added, reordered",
    ):
        worker.validate_coordinate_archive_for_block(
            archive, items, expected_role=worker.ROLE_MINILM
        )


def test_archive_parser_rejects_noncanonical_coordinate_and_policy_drift() -> None:
    mini, _ce, _mini_fake, _ce_fake = _run_both(_block(pairs=3))
    payload = copy.deepcopy(mini.payload())
    payload["rows"][0]["coordinate_float64_hex"] = "0X1.0P+0"  # type: ignore[index]
    _rehash(payload, "archive_sha256")
    with pytest.raises(worker.MmqaP1BlockCoordinateWorkerError, match="float64"):
        worker.parse_coordinate_archive_payload(payload)

    payload = copy.deepcopy(mini.payload())
    payload["dynamic_batch_resize_count"] = 1
    _rehash(payload, "archive_sha256")
    with pytest.raises(worker.MmqaP1BlockCoordinateWorkerError, match="policy"):
        worker.parse_coordinate_archive_payload(payload)


@pytest.mark.parametrize("stage", ["initialize", "infer"])
def test_oom_or_backend_failure_terminates_without_retry(stage: str) -> None:
    counts = {"initialize": 0, "infer": 0}

    def initialize(**_kwargs: object):
        counts["initialize"] += 1
        if stage == "initialize":
            raise RuntimeError("CUDA out of memory")

        def infer(**_batch: object) -> object:
            counts["infer"] += 1
            raise RuntimeError("CUDA out of memory")

        return infer

    with pytest.raises(
        worker.MmqaP1BlockCoordinateWorkerError, match="no .*retry|no retry"
    ):
        worker.run_block_coordinate_worker(
            _block(pairs=3),
            model_binding=_binding(worker.ROLE_MINILM),
            initialize_model=initialize,
        )
    assert counts["initialize"] == 1
    assert counts["infer"] == int(stage == "infer")


def test_merge_requires_same_block_runtime_and_forms_actions_in_block_order() -> None:
    items = _block(pairs=3)
    mini, ce, _mini_fake, _ce_fake = _run_both(items)
    result = worker.merge_coordinate_archives(
        items, minilm_archive=mini, cross_encoder_archive=ce
    )
    assert tuple(row.work_id for row in result.items) == tuple(
        item.work_id for item in items
    )
    assert all(
        row.actions.shared_closure.ordinals
        == tuple(unit.ordinal for unit in item.work_item.units)
        for row, item in zip(result.items, items, strict=True)
    )
    receipt = result.receipt()
    assert receipt["item_count"] == 2
    assert receipt["coordinate_archive_text_field_count"] == 0
    assert receipt["gold_answer_support_family_qid_read_count"] == 0
    assert receipt["retry_replay_resample_count"] == 0

    other_runtime = _FakeInitializer()
    mismatched_ce = worker.run_block_coordinate_worker(
        items,
        model_binding=_binding(worker.ROLE_CROSS_ENCODER, runtime="b" * 64),
        initialize_model=other_runtime,
    )
    with pytest.raises(worker.MmqaP1BlockCoordinateWorkerError, match="runtime"):
        worker.merge_coordinate_archives(
            items,
            minilm_archive=mini,
            cross_encoder_archive=mismatched_ce,
        )


def test_merge_rejects_wrong_or_partial_per_item_e5_mapping() -> None:
    items = _block(pairs=3)
    mini, ce, _mini_fake, _ce_fake = _run_both(items)
    with pytest.raises(worker.MmqaP1BlockCoordinateWorkerError, match="mapping"):
        worker.merge_coordinate_archives(
            items,
            minilm_archive=mini,
            cross_encoder_archive=ce,
            e5_models_by_work_id={items[0].work_id: None},
        )


def test_cli_uses_injected_fake_without_loading_real_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = _block(pairs=3)
    input_path = tmp_path / "block.private.json"
    output_path = tmp_path / "coordinates.private.json"
    _write_canonical(input_path, worker.anonymous_block_payload(items))
    fake = _FakeInitializer()
    monkeypatch.setattr(worker, "_production_initializer", fake)

    result = worker.main(
        [
            "--role",
            worker.ROLE_MINILM,
            "--input-block",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            "/frozen/models/all-MiniLM-L6-v2",
            "--required-tree-sha256",
            executor.MINILM_REQUIRED_TREE_SHA256,
            "--local-runtime-identity-sha256",
            "a" * 64,
        ]
    )
    assert result == 0
    assert len(fake.initializations) == 1
    assert output_path.exists()
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    archive = worker.load_coordinate_archive(output_path)
    worker.validate_coordinate_archive_for_block(
        archive, items, expected_role=worker.ROLE_MINILM
    )
