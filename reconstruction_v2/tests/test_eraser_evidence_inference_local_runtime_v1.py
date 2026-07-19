from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
from typing import Any, Iterator, Sequence

import numpy as np
import pytest

from assumption_agent.benchmarks import (
    eraser_evidence_inference_local_runtime_v1 as subject,
)


def _view(
    marker: str,
    *,
    sentence_count: int = 8,
    query: str | None = None,
) -> subject.ItemTextView:
    return subject.ItemTextView(
        item_commitment_sha256=hashlib.sha256(marker.encode("ascii")).hexdigest(),
        query=query if query is not None else f"Exact query {marker}?",
        intervention=f"Exact intervention {marker}",
        comparator=f"Exact comparator {marker}",
        outcome=f"Exact outcome {marker}",
        official_tokenized_sentences=tuple(
            ("Exact", marker, "sentence", str(ordinal), ".")
            for ordinal in range(sentence_count)
        ),
    )


class _RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.last_matrix: np.ndarray | None = None

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        rows = tuple(texts)
        self.calls.append(rows)
        matrix = np.zeros(
            (len(rows), subject.semantic_runtime.EMBEDDING_DIMENSION),
            dtype=np.float32,
        )
        for ordinal in range(len(rows)):
            matrix[ordinal, ordinal % matrix.shape[1]] = 1.0
        self.last_matrix = matrix
        return matrix


def _prepared_path_fixture() -> subject.PreparedItemArtifact:
    view = subject.ItemTextView(
        item_commitment_sha256="e" * 64,
        query="Synthetic full query",
        intervention="Synthetic intervention",
        comparator="Synthetic comparator",
        outcome="Synthetic outcome",
        official_tokenized_sentences=tuple((f"sentence-{index}",) for index in range(12)),
    )
    units = subject._build_units(view)
    facets = subject.operator.make_official_ico_facets(
        intervention_sha256="a" * 64,
        comparator_sha256="b" * 64,
        outcome_sha256="c" * 64,
    )
    anchor_row = tuple(100 if ordinal == 8 else 0 for ordinal in range(12))
    tensor = subject.operator.make_query_semantic_tensor(
        query_sha256="d" * 64,
        facets=facets,
        facet_similarity_ints=(anchor_row, anchor_row, anchor_row),
        dense_relevance_ints=(
            100,
            99,
            98,
            97,
            96,
            -100,
            -100,
            -100,
            -100,
            -100,
            95,
            -101,
        ),
    )
    graph = subject.operator.build_query_anchored_graph(
        units=units, semantic_tensor=tensor
    )
    embeddings = np.zeros(
        (12, subject.semantic_runtime.EMBEDDING_DIMENSION), dtype=np.float32
    )
    embeddings[np.arange(12), np.arange(12)] = 1.0
    embeddings.setflags(write=False)
    return subject.PreparedItemArtifact(
        block="A_form",
        view=view,
        units=units,
        semantic_tensor=tensor,
        graph=graph,
        sentence_embeddings=embeddings,
    )


def _patch_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(subject.Path, "home", classmethod(lambda _cls: home))


@pytest.fixture
def private_tmp_path() -> Iterator[Path]:
    # The Codex App may point pytest's normal tmp_path at drvfs, which cannot
    # represent mode 0700.  Hippo's private-parent contract must be exercised
    # on the WSL filesystem instead of weakening that production check.
    with tempfile.TemporaryDirectory(
        prefix="eraser-local-runtime-v1-", dir="/tmp"
    ) as folder:
        yield Path(folder)


def test_item_text_view_preserves_exact_query_ico_tokens_and_has_no_cap() -> None:
    view = _view("long", sentence_count=257, query="  Exact query bytes?  ")

    assert view.query == "  Exact query bytes?  "
    assert view.sentence_count == 257
    assert view.sentence_texts[0] == "Exact long sentence 0 ."
    assert view.sentence_texts[-1] == "Exact long sentence 256 ."
    assert tuple(field.name for field in fields(subject.ItemTextView)) == (
        "item_commitment_sha256",
        "query",
        "intervention",
        "comparator",
        "outcome",
        "official_tokenized_sentences",
    )


@pytest.mark.parametrize(
    "change,match",
    (
        ({"query": "   "}, "query"),
        ({"item_commitment_sha256": "A" * 64}, "SHA-256"),
        ({"official_tokenized_sentences": (("only",),) * 4}, "at least five"),
        (
            {
                "official_tokenized_sentences": (
                    ("valid",),
                    ("valid",),
                    ("",),
                    ("valid",),
                    ("valid",),
                )
            },
            "invalid exact token",
        ),
    ),
)
def test_item_text_view_fails_closed(change: dict[str, object], match: str) -> None:
    with pytest.raises(subject.EraserEvidenceInferenceLocalRuntimeError, match=match):
        replace(_view("invalid"), **change)


def test_one_cross_block_encode_is_exact_complete_and_prepares_no_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _view("A", sentence_count=257, query="  A exact query  ")
    second = _view("F", sentence_count=7)
    encoder = _RecordingEncoder()
    action_calls: list[object] = []

    def forbidden_actions(**kwargs: object) -> object:
        action_calls.append(kwargs)
        raise AssertionError("preparation must not execute an action")

    monkeypatch.setattr(subject.operator, "run_all_actions", forbidden_actions)
    # Deliberately reverse mapping insertion order; block order remains frozen.
    batch = subject.prepare_semantic_batch(
        items_by_block={"F_search": (second,), "A_form": (first,)},
        encoder=encoder,
    )

    assert action_calls == []
    assert batch.encoder_call_count == 1
    assert len(encoder.calls) == 1
    expected_schedule = (
        first.query,
        first.intervention,
        first.comparator,
        first.outcome,
        *first.sentence_texts,
        second.query,
        second.intervention,
        second.comparator,
        second.outcome,
        *second.sentence_texts,
    )
    assert encoder.calls[0] == expected_schedule
    assert batch.encoded_text_count == len(expected_schedule)
    assert tuple(item.block for item in batch.items) == ("A_form", "F_search")
    prepared = batch.items[0]
    assert prepared.sentence_count == 257
    assert prepared.graph.units == prepared.units
    assert prepared.units[-1].end_token == 257 * 5
    assert prepared.units[0].sentence_sha256 == hashlib.sha256(
        first.sentence_texts[0].encode("utf-8")
    ).hexdigest()
    assert prepared.semantic_tensor.query_sha256 == hashlib.sha256(
        first.query.encode("utf-8")
    ).hexdigest()
    assert prepared.sentence_embeddings.shape == (257, 384)
    assert prepared.sentence_embeddings.flags.writeable is False
    assert not hasattr(prepared, "actions")
    assert not hasattr(prepared, "sentence_cosine_int_matrix")
    serialized = json.loads(json.dumps(batch.binding_payload()))
    assert serialized["action_execution_count"] == 0
    assert serialized["exact_text_or_embedding_persisted"] is False

    assert encoder.last_matrix is not None
    expected_dense_zero = subject.qasper_binding.quantized_cosine_similarity(
        encoder.last_matrix[0], encoder.last_matrix[4]
    )
    assert prepared.semantic_tensor.dense_relevance_ints[0] == expected_dense_zero


def test_cross_block_encoder_shape_normalization_and_duplicates_fail_closed() -> None:
    view = _view("bad")

    def wrong_shape(_texts: Sequence[str]) -> np.ndarray:
        return np.ones((1, 384), dtype=np.float32)

    with pytest.raises(subject.EraserEvidenceInferenceLocalRuntimeError, match="shape"):
        subject.prepare_semantic_batch(
            items_by_block={"A_form": (view,)}, encoder=wrong_shape
        )

    def unnormalized(texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 384), dtype=np.float32)

    with pytest.raises(subject.EraserEvidenceInferenceLocalRuntimeError, match="normalized"):
        subject.prepare_semantic_batch(
            items_by_block={"A_form": (view,)}, encoder=unnormalized
        )

    with pytest.raises(subject.EraserEvidenceInferenceLocalRuntimeError, match="duplicated"):
        subject.prepare_semantic_batch(
            items_by_block={"A_form": (view,), "F_search": (view,)},
            encoder=_RecordingEncoder(),
        )


def test_agent_and_raw_actions_start_only_inside_independent_logical_futures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_path_fixture()
    original_all = subject.operator.run_all_actions
    all_calls: list[object] = []

    def counted_all(**kwargs: object) -> object:
        all_calls.append(kwargs)
        return original_all(**kwargs)  # type: ignore[arg-type]

    pair_calls: list[tuple[object, object]] = []
    original_cosine = subject.qasper_binding.quantized_cosine_similarity

    def counted_cosine(left: object, right: object) -> int:
        pair_calls.append((left, right))
        return original_cosine(left, right)

    monkeypatch.setattr(subject.operator, "run_all_actions", counted_all)
    monkeypatch.setattr(
        subject.qasper_binding, "quantized_cosine_similarity", counted_cosine
    )
    agent = subject.execute_agent(prepared)

    assert len(all_calls) == 1
    assert agent.r0_action.output_top5 == (0, 1, 2, 3, 4)
    assert agent.r7_action.output_top5 == (10, 8, 7, 9, 6)
    assert len(agent.pair_rows) == len(pair_calls) == 20
    assert tuple((left, right) for left, right, _value in agent.pair_rows) == tuple(
        sorted(
            set(subject._canonical_pair_union((0, 1, 2, 3, 4), (10, 8, 7, 9, 6)))
        )
    )
    assert all(value == 0 for _left, _right, value in agent.pair_rows)
    agent_payload = json.loads(json.dumps(agent.payload()))
    assert agent_payload["selected_pair_count"] == 20
    assert agent_payload["full_square_pair_scan_performed"] is False

    original_raw = subject.operator.run_action
    raw_calls: list[object] = []

    def counted_raw(**kwargs: object) -> object:
        raw_calls.append(kwargs)
        return original_raw(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subject.operator, "run_action", counted_raw)
    raw = subject.execute_raw(prepared)
    assert len(raw_calls) == 1
    assert raw.top5 == agent.r0_action.output_top5
    assert raw.r0_action is not agent.r0_action
    raw_payload = json.loads(json.dumps(raw.payload()))
    assert raw_payload["independent_r0_execution"] is True

    hippo_payload = json.loads(
        json.dumps(
            subject.HippoExecutionArtifact(
                block="A_form",
                item_commitment_sha256=prepared.item_commitment_sha256,
                top5=(0, 1, 2, 3, 4),
            ).payload()
        )
    )
    assert hippo_payload["top5"] == [0, 1, 2, 3, 4]
    assert hippo_payload["exact_text_or_index_persisted"] is False


def test_prepared_embeddings_are_owned_read_only_and_fail_closed_on_mutation() -> None:
    prepared = _prepared_path_fixture()
    with pytest.raises(ValueError):
        prepared.sentence_embeddings[0, 0] = 0.0

    writable = prepared.sentence_embeddings.copy()
    with pytest.raises(
        subject.EraserEvidenceInferenceLocalRuntimeError,
        match="immutable normalized float32",
    ):
        replace(prepared, sentence_embeddings=writable)


def test_preflight_binds_exact_assets_with_zero_model_source_and_network_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    manifests = project / "manifests"
    manifests.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    (manifests / subject.MINILM_ASSET_RELATIVE.name).write_bytes(
        (repository / subject.MINILM_ASSET_RELATIVE).read_bytes()
    )
    home = tmp_path / "home"
    _patch_home(monkeypatch, home)
    config = subject.default_formal_runtime_config(project)
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_minilm(asset: object, *, snapshot_root: Path) -> dict[str, object]:
        calls.append(("minilm", {"asset": asset, "snapshot_root": snapshot_root}))
        return {
            "runtime_asset_manifest_hash": subject.MINILM_ASSET_MANIFEST_SHA256,
            "snapshot_revision": subject.MINILM_SNAPSHOT_REVISION,
            "weights_sha256": "f" * 64,
        }

    def fake_hippo(**kwargs: Any) -> dict[str, object]:
        calls.append(("hippo", kwargs))
        return {
            "attestation_receipt_sha256": (
                subject.HIPPORAG_ATTESTATION_RECEIPT_SHA256
            )
        }

    monkeypatch.setattr(subject.semantic_runtime, "verify_runtime_asset", fake_minilm)
    monkeypatch.setattr(subject, "verify_formal_runtime_attestation_v3", fake_hippo)
    receipt = subject.preflight_formal_runtime_config(config)

    assert [name for name, _kwargs in calls] == ["minilm", "hippo"]
    assert calls[0][1]["snapshot_root"] == config.minilm_snapshot_root
    assert calls[1][1]["attestation_receipt_path"] == config.hippo_attestation_receipt
    assert receipt["model_inference_calls"] == 0
    assert receipt["benchmark_source_or_private_pack_reads"] == 0
    assert receipt["external_network_calls"] == 0
    assert not config.hippo_stage_parent_root.exists()


def test_preflight_and_gateway_paths_fail_closed_before_runtime_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _patch_home(monkeypatch, tmp_path / "home")
    config = subject.default_formal_runtime_config(project)

    with pytest.raises(subject.EraserEvidenceInferenceLocalRuntimeError, match="canonical"):
        subject.preflight_formal_runtime_config(
            replace(
                config,
                minilm_asset_manifest=project / "manifests/alternate.json",
            )
        )

    config.hippo_stage_parent_root.mkdir(parents=True)
    with pytest.raises(subject.EraserEvidenceInferenceLocalRuntimeError, match="already exists"):
        subject.preflight_formal_runtime_config(config)
    gateway = subject.OfficialHippoGateway(config)
    with pytest.raises(subject.EraserEvidenceInferenceLocalRuntimeError, match="freshly"):
        gateway.prepare_blocks(("A_form",))


def test_official_hippo_gateway_prebuilds_private_blocks_and_allocates_fresh_roots(
    private_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = private_tmp_path
    project = tmp_path / "project"
    project.mkdir()
    _patch_home(monkeypatch, tmp_path / "home")
    config = subject.default_formal_runtime_config(project)
    gateway = subject.OfficialHippoGateway(config)
    with pytest.raises(subject.EraserEvidenceInferenceLocalRuntimeError, match="prebuilt"):
        gateway.retrieve(block="A_form", view=_view("early"))

    parents = gateway.prepare_blocks(("F_search", "A_form"))
    assert parents == (
        config.hippo_stage_parent_root / "A_form",
        config.hippo_stage_parent_root / "F_search",
    )
    assert stat.S_IMODE(config.hippo_stage_parent_root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in parents)

    observed: list[dict[str, Any]] = []
    observed_lock = threading.Lock()

    def fake_item_adapter(**kwargs: Any) -> tuple[int, ...]:
        assert not os.path.lexists(kwargs["work_root"])
        with observed_lock:
            observed.append(kwargs)
        return (0, 1, 2, 3, 4)

    monkeypatch.setattr(
        subject.hippo_adapter,
        "run_item_local_official_hipporag_v1",
        fake_item_adapter,
    )
    views = tuple(_view(f"thread-{index}") for index in range(24))
    with ThreadPoolExecutor(max_workers=24) as executor:
        results = tuple(
            executor.map(
                lambda view: gateway.retrieve(block="A_form", view=view),
                views,
            )
        )

    assert results == ((0, 1, 2, 3, 4),) * len(views)
    roots = [row["work_root"] for row in observed]
    assert len(roots) == len(set(roots)) == len(views)
    assert all(root.parent == parents[0] for root in roots)
    assert all(not os.path.lexists(root) for root in roots)
    by_query = {row["query"]: row for row in observed}
    for view in views:
        row = by_query[view.query]
        assert row["sentence_texts"] == view.sentence_texts
        assert row["timeout_seconds"] == 900


def test_open_runtime_uses_semantic_assignment_encoder_exact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _patch_home(monkeypatch, tmp_path / "home")
    config = subject.default_formal_runtime_config(project)
    observed: list[dict[str, Path]] = []

    class FakeEncoder:
        def __init__(self, **kwargs: Path) -> None:
            observed.append(kwargs)

        def __call__(self, texts: Sequence[str]) -> np.ndarray:
            return np.empty((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(subject.semantic_runtime, "OfflineMiniLMEncoder", FakeEncoder)
    bundle = subject.open_runtime(config)

    assert observed == [
        {
            "runtime_asset_path": config.minilm_asset_manifest,
            "snapshot_root": config.minilm_snapshot_root,
        }
    ]
    assert isinstance(bundle.encoder, FakeEncoder)
    assert isinstance(bundle.hippo, subject.OfficialHippoGateway)
