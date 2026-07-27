from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Mapping

import numpy as np
import pytest

from assumption_agent.benchmarks import dstc9_p1_formal_controller_v1 as ctl
from assumption_agent.benchmarks import dstc9_p1_formal_source_v1 as source
from assumption_agent.benchmarks import dstc9_p1_typed_core_v1 as core
from replication_runtime.dstc9_official_hipporag_v1 import contract as hippo_contract
from replication_runtime.dstc9_p1_formal_v1 import runner
from replication_runtime.qasper_minilm_portable_v2.binding import (
    PORTABLE_CANARY_SCHEMA,
    PortableOfflineMiniLMEncoder,
)
from replication_runtime.qasper_minilm_v1 import binding as frozen_minilm_v1


@pytest.fixture
def tmp_path() -> Path:
    """Use a real Linux filesystem because the runtime audits POSIX modes."""

    root = Path(tempfile.mkdtemp(prefix="dstc9-formal-runtime-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _work(index: int) -> str:
    return "dstc9-work-v1-" + hashlib.sha256(
        f"work-{index}".encode("ascii")
    ).hexdigest()


def _item(index: int, block: str) -> ctl.FormalItemView:
    return ctl.FormalItemView(
        work_id=_work(index),
        block=block,
        history=(
            core.DialogueTurn(
                speaker="U",
                text=f"public query number {index}",
            ),
        ),
    )


def _corpus() -> ctl.CorpusView:
    return ctl.CorpusView.create(
        tuple(
            core.KnowledgeSnippet(
                ordinal=index,
                entity_name=f"entity {index % 7}",
                title=f"title {index % 11}",
                body=f"body {index % 13}",
            )
            for index in range(ctl.CORPUS_SIZE)
        )
    )


def _config(root: Path) -> runner.FormalRuntimeConfig:
    absent = root / "intentionally-absent"
    return runner.FormalRuntimeConfig(
        formal_root=root / "attempt",
        p0_receipt_path=absent / "p0.json",
        private_eligibility_manifest_path=absent / "private.json",
        bundle_path=absent / "bundle.tar",
        execution_binding_sha256="1" * 64,
        coordinate_runtime_python=absent / "coordinate-python",
        coordinate_project_root=absent / "coordinate-project",
        minilm_asset_manifest=absent / "minilm-asset.json",
        minilm_model_root=absent / "minilm-model",
        cross_encoder_model_root=absent / "cross-encoder",
        hippo_runtime_python=absent / "hippo-python",
        hippo_worker_project_root=absent / "formal-worker-project",
        hippo_llm_model_root=absent / "llm-model",
        hippo_embedding_model_root=absent / "embedding-model",
        hippo_runtime_fingerprint_path=absent / "runtime.json",
        current_hardware_binding_path=absent / "hardware.json",
        current_hardware_binding_file_sha256="2" * 64,
        current_hardware_binding_self_sha256="3" * 64,
        source_free_canary_receipt_path=absent / "canary.json",
        source_free_canary_receipt_file_sha256="4" * 64,
        source_free_canary_receipt_self_sha256="5" * 64,
    )


def _canary_config(root: Path) -> runner.CanaryRuntimeConfig:
    absent = root / "intentionally-absent"
    canary_root = root / "canary-attempt"
    return runner.CanaryRuntimeConfig(
        canary_root=canary_root,
        current_hardware_binding_path=(
            canary_root / runner.CURRENT_HARDWARE_BINDING_FILENAME
        ),
        hardware_capture_id="dstc9-canary-test-v1",
        canary_binding_sha256="6" * 64,
        coordinate_runtime_python=absent / "coordinate-python",
        coordinate_project_root=absent / "coordinate-project",
        minilm_asset_manifest=absent / "minilm-asset.json",
        minilm_model_root=absent / "minilm-model",
        cross_encoder_model_root=absent / "cross-encoder",
        hippo_runtime_python=absent / "hippo-python",
        hippo_worker_project_root=absent / "formal-worker-project",
        hippo_llm_model_root=absent / "llm-model",
        hippo_embedding_model_root=absent / "embedding-model",
        hippo_runtime_fingerprint_path=absent / "runtime.json",
    )


class _BucketEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts):
        rows = tuple(texts)
        self.calls.append(rows)
        matrix = np.zeros((len(rows), runner.EMBEDDING_DIMENSION), dtype=np.float32)
        if rows == runner.PREDICTOR_PROTOTYPES:
            for index in range(4):
                matrix[index, index] = 1.0
        else:
            for index in range(len(rows)):
                matrix[index, 2] = 1.0
        return matrix


def test_public_prototype_predictor_has_only_query_items_and_frozen_tie_break():
    encoder = _BucketEncoder()
    predictor = runner.PublicPrototypeBucketPredictor(encoder)
    item = _item(1, "A_form")
    rows = predictor.predict((item,))
    assert len(rows) == 1
    assert rows[0].predicted_bucket == 2
    assert rows[0].predictor_commitment == runner.PREDICTOR_COMMITMENT
    assert encoder.calls == [
        runner.PREDICTOR_PROTOTYPES,
        (core.serialize_model_query(item.history),),
    ]
    assert tuple(runner.PublicPrototypeBucketPredictor.predict.__annotations__) == (
        "items",
        "return",
    )


def test_predictor_factory_uses_portable_v2_and_not_v1_output_hash_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[Path, Path]] = []

    class FakePortable:
        def __init__(self, *, asset_manifest_path, model_root):
            calls.append(
                (Path(asset_manifest_path), Path(model_root))
            )

        def encode(self, texts):
            rows = tuple(texts)
            matrix = np.zeros(
                (len(rows), runner.EMBEDDING_DIMENSION),
                dtype=np.float32,
            )
            for index in range(len(rows)):
                matrix[index, index % len(runner.PREDICTOR_PROTOTYPES)] = 1.0
            return matrix

    def forbidden(*_args, **_kwargs):
        raise AssertionError("v1 fixed-output oracle must not be invoked")

    monkeypatch.setattr(
        runner, "PortableOfflineMiniLMEncoder", FakePortable
    )
    monkeypatch.setattr(
        frozen_minilm_v1, "OfflineMiniLMEncoder", forbidden
    )
    monkeypatch.setattr(
        frozen_minilm_v1, "run_synthetic_canary", forbidden
    )
    predictor = runner.PublicPrototypeBucketPredictor.from_paths(
        asset_manifest_path=tmp_path / "manifest.json",
        model_root=tmp_path / "model",
    )
    assert isinstance(predictor, runner.PublicPrototypeBucketPredictor)
    assert calls == [
        (tmp_path / "manifest.json", tmp_path / "model")
    ]
    assert runner.__dict__.get("OfflineMiniLMEncoder") is None
    assert PortableOfflineMiniLMEncoder.__module__.endswith(
        "qasper_minilm_portable_v2.binding"
    )
    startup = runner.PREDICTOR_COMMITMENT_PAYLOAD[
        "portable_startup_acceptance"
    ]
    assert startup["schema"] == PORTABLE_CANARY_SCHEMA
    assert (
        startup[
            "expected_output_hash_or_allowlist_is_acceptance_oracle"
        ]
        is False
    )
    assert startup["observed_output_hashes_are_normative"] is False
    assert startup["repeat_count"] == 2


def test_coordinate_lane_primes_one_initial_batch_and_one_conditional_m_batch(
    tmp_path: Path,
):
    corpus = _corpus()
    lane = runner.CoordinateScorerLane(
        runtime_python=tmp_path / "python",
        project_root=tmp_path / "project",
        minilm_asset_manifest=tmp_path / "asset",
        minilm_model_root=tmp_path / "model",
        cross_encoder_model_root=tmp_path / "ce",
        lane_root=tmp_path / "coordinate",
        timeout_seconds=10,
        run_callable=lambda **_kwargs: {},
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_execute(_corpus, items, *, stage_name):
        calls.append((stage_name, tuple(item.work_id for item in items)))
        return {item.work_id: object() for item in items}

    lane._execute = fake_execute  # type: ignore[method-assign]
    blocks: OrderedDict[str, ctl.BlockView] = OrderedDict()
    offset = 0
    for block in ctl.INITIAL_BLOCKS:
        count = ctl.BLOCK_COUNTS[block]
        items = tuple(_item(offset + index, block) for index in range(count))
        offset += count
        blocks[block] = ctl.BlockView.create(block, items)
    lane.prime_initial(corpus, blocks)
    assert calls[0][0] == "initial_176"
    assert len(calls[0][1]) == 176
    for block in ctl.INITIAL_BLOCKS:
        assert len(lane.score(corpus, blocks[block].items)) == ctl.BLOCK_COUNTS[block]
    assert len(calls) == 1

    m_items = tuple(_item(1000 + index, "M_search") for index in range(48))
    assert len(lane.score(corpus, m_items)) == 48
    assert len(lane.score(corpus, m_items)) == 48
    assert [name for name, _ in calls] == ["initial_176", "M_search_48"]


def test_official_hippo_lane_builds_once_and_reopens_for_two_blocks(
    tmp_path: Path,
):
    corpus = _corpus()
    calls: list[tuple[str, object]] = []

    def build(**kwargs):
        calls.append(("build", kwargs["worker_project_root"]))
        return {"status": "passed"}

    def retrieve(**kwargs):
        queries = hippo_contract.validate_query_input(kwargs["query_input"])
        calls.append(("retrieve", tuple(row.work_id for row in queries.queries)))
        return hippo_contract.RetrievalBatch(
            indices=tuple((0, 1, 2, 3, 4) for _ in queries.queries),
            receipt={"receipt_sha256": "a" * 64},
        )

    lane = runner.OfficialHippoLane(
        runtime_python=tmp_path / "python",
        worker_project_root=tmp_path / "worker-project",
        current_hardware_binding_path=tmp_path / "hardware.json",
        local_llm_model=tmp_path / "llm",
        local_embedding_model=tmp_path / "embed",
        runtime_fingerprint_path=tmp_path / "fingerprint",
        lane_root=tmp_path / "hippo",
        build_timeout_seconds=10,
        retrieve_timeout_seconds=10,
        build_callable=build,
        retrieve_callable=retrieve,
    )
    try:
        lane.start_build(corpus)
        a = lane.retrieve(corpus, (_item(1, "A_hold"),))
        m = lane.retrieve(corpus, (_item(2, "M_search"),))
        assert a[0].top5_ordinals == (0, 1, 2, 3, 4)
        assert m[0].top5_ordinals == (0, 1, 2, 3, 4)
        assert lane.build_call_count == 1
        assert lane.retrieve_call_count == 2
        assert set(lane.private_retrieval_commitments) == {
            "A_hold",
            "M_search",
        }
        for block in ("A_hold", "M_search"):
            evidence = tmp_path / "hippo" / f"{block}.retrieval.private.json"
            assert evidence.is_file()
            assert evidence.stat().st_mode & 0o777 == 0o400
        with pytest.raises(runner.Dstc9P1FormalRuntimeError, match="lifecycle"):
            lane.retrieve(corpus, (_item(3, "A_hold"),))
    finally:
        lane.close()
    assert calls[0] == ("build", tmp_path / "worker-project")
    assert [name for name, _ in calls] == ["build", "retrieve", "retrieve"]


def _write_source_artifact(
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int,
    row_count: int,
) -> dict[str, object]:
    raw = source.canonical_bytes(value, newline=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "mode": f"{mode:04o}",
        "row_count": row_count,
        "self_sha256": value["self_sha256"],
        "size_bytes": len(raw),
    }


def _source_fixture(tmp_path: Path):
    corpus = _corpus()
    source_root = tmp_path / "source"
    source_root.mkdir()
    outputs = runner._source_output_paths(source_root)

    corpus_value = source.self_hashed(
        {
            "schema": source.PUBLIC_CORPUS_SCHEMA,
            "snippets": [
                core.snippet_public_payload(row) for row in corpus.snippets
            ],
            "study_id": ctl.STUDY_ID,
            "version": source.VERSION,
        }
    )
    corpus_binding = _write_source_artifact(
        outputs.public_corpus,
        corpus_value,
        mode=0o600,
        row_count=ctl.CORPUS_SIZE,
    )
    public_bindings: dict[str, object] = {}
    qrel_bindings: dict[str, object] = {}
    next_index = 50
    for block in source.BLOCKS:
        selected = tuple(
            _item(next_index + offset, block)
            for offset in range(ctl.BLOCK_COUNTS[block])
        )
        next_index += len(selected)
        value = source.self_hashed(
            {
                "block_id": block,
                "items": [
                    {
                        "history": [
                            core.turn_public_payload(turn)
                            for turn in item.history
                        ],
                        "normalized_query_sha256": (
                            core.normalized_query_sha256(item.history)
                        ),
                        "work_id": item.work_id,
                    }
                    for item in selected
                ],
                "schema": source.PUBLIC_BLOCK_SCHEMA,
                "study_id": ctl.STUDY_ID,
                "version": source.VERSION,
            }
        )
        public_bindings[block] = _write_source_artifact(
            outputs.public_blocks()[block],
            value,
            mode=0o400 if block == "M_search" else 0o600,
            row_count=len(selected),
        )
        if block in source.QREL_BLOCKS:
            family_width = ctl.FAMILY_COUNTS[block]
            qrel = source.self_hashed(
                {
                    "block_id": block,
                    "qrels": [
                        {
                            "family": ctl.FAMILIES[
                                index // family_width
                            ],
                            "gold_ordinal": 0,
                            "work_id": item.work_id,
                        }
                        for index, item in enumerate(selected)
                    ],
                    "schema": source.PRIVATE_QREL_SCHEMA,
                    "study_id": ctl.STUDY_ID,
                    "version": source.VERSION,
                }
            )
            qrel_bindings[block] = _write_source_artifact(
                outputs.private_qrels()[block],
                qrel,
                mode=0o400,
                row_count=len(selected),
            )
    receipt = source.self_hashed(
        {
            "artifact_binding": {
                "private_qrels": qrel_bindings,
                "public_blocks": public_bindings,
                "public_corpus": corpus_binding,
            },
            "disjointness_aggregate": {"overlap": 0},
            "p0_binding": {"receipt": "safe"},
            "quota": {"frozen": True},
            "schema": source.SELECTION_RECEIPT_SCHEMA,
            "selection": {"seed": source.SELECTION_SEED},
            "source_access": {"formal_source_access_count": 1},
            "status": "selected_and_sealed",
            "study_id": ctl.STUDY_ID,
        }
    )
    outputs.safe_selection_receipt.write_bytes(
        source.canonical_bytes(receipt, newline=True)
    )
    outputs.safe_selection_receipt.chmod(0o600)
    return outputs, receipt


class _CoordinatePrimeStub:
    def __init__(self):
        self.calls = []

    def prime_initial(self, corpus, blocks):
        self.calls.append((corpus, tuple(blocks)))


class _HippoStartStub:
    def __init__(self):
        self.calls = []

    def start_build(self, corpus):
        self.calls.append(corpus)


def _action_archive(path: Path, block: str) -> Mapping[str, object]:
    value = ctl.self_hashed(
        {
            "block": block,
            "rows": [],
            "schema": f"test_{block}_action_archive",
            "study_id": ctl.STUDY_ID,
        }
    )
    path.write_bytes(ctl.canonical_bytes(value))
    path.chmod(0o400)
    return value


def test_acquisition_never_opens_qrels_or_m_before_exact_authorization(
    tmp_path: Path,
):
    outputs, receipt = _source_fixture(tmp_path)
    controller_root = tmp_path / "controller"
    controller_root.mkdir()
    coordinate = _CoordinatePrimeStub()
    hippo = _HippoStartStub()
    boundary = runner.SealedSourceAcquisitionBoundary(
        outputs=outputs,
        selection_receipt=receipt,
        controller_root=controller_root,
        coordinate_lane=coordinate,  # type: ignore[arg-type]
        hippo_lane=hippo,  # type: ignore[arg-type]
    )
    claim = boundary.claim_formal_attempt("b" * 64)
    corpus = boundary.load_public_corpus(claim)
    assert hippo.calls == [corpus]

    for block in ctl.INITIAL_BLOCKS:
        boundary.load_label_free_block(block, None)
    assert coordinate.calls[0][1] == ctl.INITIAL_BLOCKS
    assert boundary.qrel_open_count == {
        "A_form": 0,
        "A_hold": 0,
        "M_search": 0,
    }
    with pytest.raises(runner.Dstc9P1FormalRuntimeError, match="authorization"):
        boundary.load_label_free_block("M_search", None)
    assert boundary.public_open_count["M_search"] == 0

    missing = controller_root / "A_form.actions.private.json"
    with pytest.raises(runner.Dstc9P1FormalRuntimeError, match="sealed action"):
        boundary.release_qrels_after_action_seal("A_form", missing, {})
    assert boundary.qrel_open_count["A_form"] == 0
    action = _action_archive(missing, "A_form")
    pack = boundary.release_qrels_after_action_seal(
        "A_form", missing, action
    )
    assert pack.action_archive_sha256 == action["self_sha256"]
    assert boundary.qrel_open_count["A_form"] == 1

    authorization = ctl.self_hashed(
        {
            "A_hold_E1_minus_E0": {
                "item_count": 48,
                "negative_count": 0,
                "net_utility": 1,
                "one_sided_exact_magnitude_preserving_tail": {
                    "denominator": 16,
                    "numerator": 1,
                },
                "positive_count": 1,
                "tie_count": 47,
            },
            "block_disjointness_commitment": (
                claim.block_disjointness_commitment
            ),
            "comparison_net_strictly_positive": True,
            "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth": True,
            "schema": (
                f"{ctl.VERSION}_"
                "M_search_materialization_authorization_v1"
            ),
            "status": "A_hold_E1_promoted",
            "study_id": ctl.STUDY_ID,
        }
    )
    authorization_path = (
        controller_root / ctl.PROMOTION_AUTHORIZATION_FILENAME
    )
    with pytest.raises(runner.Dstc9P1FormalRuntimeError, match="authorization"):
        boundary.load_label_free_block("M_search", authorization)
    bad_body = dict(authorization)
    bad_body.pop("self_sha256")
    bad_body["A_hold_E1_minus_E0"] = {
        **bad_body["A_hold_E1_minus_E0"],
        "one_sided_exact_magnitude_preserving_tail": {
            "denominator": 2,
            "numerator": 1,
        },
    }
    bad_authorization = ctl.self_hashed(bad_body)
    authorization_path.write_bytes(ctl.canonical_bytes(bad_authorization))
    authorization_path.chmod(0o400)
    with pytest.raises(runner.Dstc9P1FormalRuntimeError, match="authorization"):
        boundary.load_label_free_block("M_search", bad_authorization)
    authorization_path.unlink()
    authorization_path.write_bytes(ctl.canonical_bytes(authorization))
    authorization_path.chmod(0o400)
    boundary.load_label_free_block("M_search", authorization)
    m_path = controller_root / "M_search.actions.private.json"
    m_action = _action_archive(m_path, "M_search")
    boundary.release_qrels_after_action_seal(
        "M_search", m_path, m_action
    )
    assert boundary.public_open_count["M_search"] == 1
    assert boundary.qrel_open_count["M_search"] == 1


def test_compile_failure_writes_safe_terminal_before_controller(
    tmp_path: Path,
):
    config = _config(tmp_path)
    controller_called = False

    def fail_compile(**_kwargs):
        raise source.Dstc9P1FormalSourceError(
            "synthetic_compile_failure", "no source result"
        )

    def controller(**_kwargs):
        nonlocal controller_called
        controller_called = True
        return {}

    with pytest.raises(runner.Dstc9P1FormalRuntimeError):
            runner.run_formal_study_once(
                config,
                compile_callable=fail_compile,
                controller_callable=controller,
                preformal_verifier=lambda _config: {
                    "canary_receipt_self_sha256": "5" * 64,
                    "hardware_binding_self_sha256": "3" * 64,
                },
            )
    assert controller_called is False
    terminal = json.loads(
        (config.formal_root / runner.OUTER_TERMINAL).read_text()
    )
    assert terminal["status"] == "terminal_formal_failure_no_retry"
    assert terminal["failure_stage"] == "compile_formal_source_once"
    assert terminal["online_or_API_evaluator_calls"] == 0
    assert terminal["current_hardware_binding_self_sha256"] == "3" * 64


def test_live_hardware_failure_happens_before_compile_source(tmp_path: Path):
    config = _config(tmp_path)
    compile_called = False

    def compile_source(**_kwargs):
        nonlocal compile_called
        compile_called = True
        return {}

    def fail_hardware(_config):
        raise runner.Dstc9P1FormalRuntimeError("live hardware changed")

    with pytest.raises(
        runner.Dstc9P1FormalRuntimeError, match="live hardware changed"
    ):
        runner.run_formal_study_once(
            config,
            compile_callable=compile_source,
            preformal_verifier=fail_hardware,
        )
    assert compile_called is False
    terminal = json.loads(
        (config.formal_root / runner.OUTER_TERMINAL).read_text()
    )
    assert terminal["failure_stage"] == (
        "verify_live_hardware_and_source_free_canary_before_source"
    )


def test_preformal_verifier_requires_frozen_predictor_canary_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    hardware = {"study_id": ctl.STUDY_ID}
    canary = {
        "schema": runner.CANARY_SCHEMA,
        "status": "passed_source_free_two_lane_canary_once",
        "study_id": ctl.STUDY_ID,
        "formal_source_access_count": 0,
        "current_hardware_binding_file_sha256": (
            config.current_hardware_binding_file_sha256
        ),
        "current_hardware_binding_self_sha256": (
            config.current_hardware_binding_self_sha256
        ),
        "predictor_count": 1,
        "predictor_commitment_sha256": runner.PREDICTOR_COMMITMENT,
        "predictor_result_commitment_sha256": "a" * 64,
    }

    def load_receipt(_path, *, field, **_kwargs):
        return hardware if field == "current hardware binding" else canary

    monkeypatch.setattr(runner, "_load_exact_safe_receipt", load_receipt)
    monkeypatch.setattr(
        runner.runtime_binding,
        "verify_current_study_hardware_binding",
        lambda **_kwargs: {
            "receipt_file_sha256": (
                config.current_hardware_binding_file_sha256
            ),
            "receipt_self_sha256": (
                config.current_hardware_binding_self_sha256
            ),
        },
    )
    result = runner._verify_preformal_hardware_and_canary(config)
    assert result["canary_receipt_self_sha256"] == (
        config.source_free_canary_receipt_self_sha256
    )
    canary.pop("predictor_count")
    with pytest.raises(
        runner.Dstc9P1FormalRuntimeError,
        match="canary receipt binding",
    ):
        runner._verify_preformal_hardware_and_canary(config)


def test_source_free_canary_uses_two_concurrent_lanes_and_no_source(
    tmp_path: Path,
):
    config = _canary_config(tmp_path)
    barrier = threading.Barrier(2, timeout=3)
    predictor_calls: list[tuple[Path, Path, str]] = []

    class FakeCoordinate:
        def __init__(self, **_kwargs):
            assert config.current_hardware_binding_path.is_file()
            self.worker_call_count = 0

        def _execute(self, _corpus, items, *, stage_name):
            assert stage_name == "synthetic_one_query"
            self.worker_call_count += 1
            barrier.wait()
            return {items[0].work_id: object()}

    class FakeHippo:
        def __init__(self, **kwargs):
            assert kwargs["worker_project_root"] == (
                config.hippo_worker_project_root
            )
            assert kwargs["current_hardware_binding_path"] == (
                config.current_hardware_binding_path
            )
            self.build_call_count = 0
            self.retrieve_call_count = 0
            self.corpus = None

        def start_build(self, corpus):
            self.build_call_count += 1
            self.corpus = corpus

        def retrieve(self, corpus, items):
            assert corpus is self.corpus
            self.retrieve_call_count += 1
            barrier.wait()
            return (
                ctl.HippoResult(
                    work_id=items[0].work_id,
                    block=items[0].block,
                    normalized_query_sha256=core.normalized_query_sha256(
                        items[0].history
                    ),
                    corpus_projection_sha256=corpus.projection_sha256,
                    top5_ordinals=(0, 1, 2, 3, 4),
                    receipt_sha256="c" * 64,
                ),
            )

        def close(self):
            return None

    class FakePredictor:
        commitment = runner.PREDICTOR_COMMITMENT

        def predict(self, items):
            assert len(items) == 1
            predictor_calls.append(
                (
                    config.minilm_asset_manifest,
                    config.minilm_model_root,
                    items[0].work_id,
                )
            )
            return (
                ctl.BucketPrediction.create(
                    item=items[0],
                    predicted_bucket=2,
                    predictor_commitment=self.commitment,
                ),
            )

    def make_predictor(*, asset_manifest_path, model_root):
        assert asset_manifest_path == config.minilm_asset_manifest
        assert model_root == config.minilm_model_root
        assert set(config.__dataclass_fields__).isdisjoint(
            {
                "p0_receipt_path",
                "private_eligibility_manifest_path",
                "bundle_path",
            }
        )
        return FakePredictor()

    hardware = runner._with_self_hash(
        {
            "capture_id": config.hardware_capture_id,
            "hardware": {"safe": True},
            "schema": "test_hardware",
            "source_free_boundary": {"formal_source_open_count": 0},
            "status": "test",
            "study_id": ctl.STUDY_ID,
        }
    )

    def verify_hardware(*, path, worker_project_root, expected_study_id):
        assert worker_project_root == config.hippo_worker_project_root
        assert expected_study_id == ctl.STUDY_ID
        raw = path.read_bytes()
        return {
            "receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
            "receipt_self_sha256": hardware["self_sha256"],
        }

    receipt = runner.run_source_free_canary_once(
        config,
        predictor_factory=make_predictor,
        coordinate_factory=FakeCoordinate,
        hippo_factory=FakeHippo,
        hardware_capture_callable=lambda **_kwargs: hardware,
        hardware_verify_callable=verify_hardware,
    )
    assert receipt["status"] == "passed_source_free_two_lane_canary_once"
    assert receipt["formal_source_access_count"] == 0
    assert receipt["synthetic_corpus_count"] == 2900
    assert receipt["synthetic_unique_serialized_text_count"] == 5
    assert receipt["predictor_count"] == 1
    assert receipt["predictor_commitment_sha256"] == (
        runner.PREDICTOR_COMMITMENT
    )
    assert len(receipt["predictor_result_commitment_sha256"]) == 64
    assert len(predictor_calls) == 1
    assert (
        config.canary_root / runner.CANARY_ATTEMPT_MARKER_FILENAME
    ).is_file()
    assert receipt["current_hardware_binding_self_sha256"] == (
        hardware["self_sha256"]
    )
    assert set(config.__dataclass_fields__).isdisjoint(
        {"p0_receipt_path", "private_eligibility_manifest_path", "bundle_path"}
    )
