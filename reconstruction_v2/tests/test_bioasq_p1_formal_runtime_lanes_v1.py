from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading

import pytest

from assumption_agent.benchmarks import (
    bioasq_p1_formal_controller_v1 as ctl,
)
from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as core
from replication_runtime.bioasq_coordinate_scorer_v1 import (
    contract as coordinate_contract,
)
from replication_runtime.bioasq_p1_formal_v1 import lanes
from replication_runtime.bioasq_p1_formal_v1.contract import (
    BioasqP1FormalRuntimeError,
)
from replication_runtime.dstc9_official_hipporag_v1 import (
    contract as hippo_contract,
)


@pytest.fixture
def tmp_path() -> Path:
    """Use a Linux filesystem because the lanes enforce exact POSIX modes."""

    root = Path(tempfile.mkdtemp(prefix="bioasq-lanes-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _work(index: int) -> str:
    return "bioasq-work-v2-" + hashlib.sha256(
        f"work-{index}".encode("ascii")
    ).hexdigest()


def _item(index: int) -> ctl.FormalItemView:
    return ctl.FormalItemView(
        work_id=_work(index),
        question_text=f"What biomedical finding is numbered {index}?",
    )


def _corpus(*, changed: bool = False) -> ctl.CorpusView:
    return ctl.CorpusView.create(
        tuple(
            core.Passage(
                ordinal=index,
                text=(
                    f"{'changed ' if changed and index == 0 else ''}"
                    f"synthetic biomedical passage {index % 17}"
                ),
            )
            for index in range(ctl.CORPUS_SIZE)
        )
    )


def _coordinate_lane(
    tmp_path: Path,
    run_callable,
    *,
    name: str = "coordinate",
) -> lanes.CoordinateScorerLane:
    absent = tmp_path / "intentionally-absent"
    return lanes.CoordinateScorerLane(
        runtime_python=absent / "python",
        project_root=absent / "project",
        minilm_asset_manifest=absent / "manifest.json",
        minilm_model_root=absent / "minilm",
        cross_encoder_model_root=absent / "cross-encoder",
        expected_model_binding_sha256="a" * 64,
        lane_root=tmp_path / name,
        timeout_seconds=30,
        run_callable=run_callable,
    )


def _valid_coordinate_output(
    input_value: object,
    *,
    call_ordinal: int,
) -> dict[str, object]:
    scorer_input = coordinate_contract.validate_input(input_value)
    zeros = (0,) * (ctl.CORPUS_SIZE - 1)
    rows = []
    for query_ordinal in range(len(scorer_input.queries)):
        # The first vector cell encodes input ordinal.  This lets the test
        # prove the lane maps query_ordinal, not output position or work ID.
        values = (call_ordinal * 1_000 + query_ordinal, *zeros)
        rows.append(
            {name: values for name in coordinate_contract.SCORE_NAMES}
        )
    return coordinate_contract.make_output(
        scorer_input=scorer_input,
        score_rows=rows,
        model_binding_sha256="a" * 64,
    )


def test_coordinate_lane_maps_ordinals_and_runs_exact_176_then_48_once(
    tmp_path: Path,
) -> None:
    corpus = _corpus()
    calls: list[tuple[int, tuple[str, ...], Path]] = []

    def run(**kwargs):
        scorer_input = coordinate_contract.validate_input(
            kwargs["input_value"]
        )
        ordinal = len(calls)
        calls.append(
            (
                len(scorer_input.queries),
                tuple(row.text for row in scorer_input.queries),
                kwargs["work_root"],
            )
        )
        return _valid_coordinate_output(
            kwargs["input_value"],
            call_ordinal=ordinal,
        )

    lane = _coordinate_lane(tmp_path, run)
    initial = tuple(
        _item(index)
        for index in range(lanes.INITIAL_COORDINATE_QUERY_COUNT)
    )
    initial_rows = lane.score(corpus, initial)
    assert len(initial_rows) == 176
    assert tuple(row.work_id for row in initial_rows) == tuple(
        item.work_id for item in initial
    )
    assert tuple(row.score_vectors["raw_ce"][0] for row in initial_rows) == (
        tuple(range(176))
    )

    # An identical request is served from the private cache.
    cached = lane.score(corpus, initial)
    assert cached == initial_rows
    assert lane.worker_call_count == 1

    m_items = tuple(_item(1_000 + index) for index in range(48))
    m_rows = lane.score(corpus, m_items)
    assert len(m_rows) == 48
    assert tuple(row.work_id for row in m_rows) == tuple(
        item.work_id for item in m_items
    )
    assert tuple(row.score_vectors["raw_ce"][0] for row in m_rows) == tuple(
        1_000 + index for index in range(48)
    )
    assert lane.score(corpus, m_items) == m_rows
    assert lane.worker_call_count == 2
    assert [count for count, _, _ in calls] == [176, 48]
    assert calls[0][1] == tuple(item.question_text for item in initial)
    assert calls[1][1] == tuple(item.question_text for item in m_items)
    assert set(lane.private_output_commitments) == {
        "initial_176",
        "M_search_48",
    }
    assert set(lane.private_receipt_commitments) == {
        "initial_176",
        "M_search_48",
    }
    for stage in ("initial_176", "M_search_48"):
        private = (
            tmp_path
            / "coordinate"
            / stage
            / "adapter_output.private.json"
        )
        assert private.is_file()
        assert private.stat().st_mode & 0o777 == 0o400
        value = json.loads(private.read_text(encoding="ascii"))
        assert "rows" in value

    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="lifecycle",
    ):
        lane.score(
            corpus,
            tuple(_item(2_000 + index) for index in range(48)),
        )
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="corpus changed",
    ):
        lane.score(_corpus(changed=True), m_items)
    assert lane.worker_call_count == 2


def test_coordinate_lane_marks_failure_attempted_and_never_retries(
    tmp_path: Path,
) -> None:
    calls = 0

    def fail(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic worker failure")

    lane = _coordinate_lane(tmp_path, fail, name="coordinate-failure")
    corpus = _corpus()
    initial = tuple(_item(index) for index in range(176))
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="failed without retry",
    ):
        lane.score(corpus, initial)
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="cannot retry",
    ):
        lane.score(corpus, initial)
    assert calls == 1
    assert lane.worker_call_count == 1


def test_coordinate_lane_rejects_model_binding_different_from_canary(
    tmp_path: Path,
) -> None:
    def drift(**kwargs):
        scorer_input = coordinate_contract.validate_input(
            kwargs["input_value"]
        )
        zeros = (0,) * ctl.CORPUS_SIZE
        return coordinate_contract.make_output(
            scorer_input=scorer_input,
            score_rows=[
                {
                    name: zeros
                    for name in coordinate_contract.SCORE_NAMES
                }
                for _ in scorer_input.queries
            ],
            model_binding_sha256="f" * 64,
        )

    lane = _coordinate_lane(tmp_path, drift, name="coordinate-drift")
    with pytest.raises(
        BioasqP1FormalRuntimeError,
        match="differs from the canary",
    ):
        lane.score(
            _corpus(),
            tuple(_item(index) for index in range(176)),
        )
    assert lane.worker_call_count == 1


def _hippo_lane(
    tmp_path: Path,
    *,
    build_callable,
    retrieve_callable,
) -> lanes.OfficialHippoLane:
    absent = tmp_path / "intentionally-absent"
    return lanes.OfficialHippoLane(
        runtime_python=absent / "python",
        worker_project_root=absent / "worker-project",
        current_hardware_binding_path=absent / "hardware.json",
        local_llm_model=absent / "llm",
        local_embedding_model=absent / "embedding",
        runtime_fingerprint_path=absent / "fingerprint.json",
        lane_root=tmp_path / "hippo",
        build_timeout_seconds=30,
        retrieve_timeout_seconds=30,
        build_callable=build_callable,
        retrieve_callable=retrieve_callable,
    )


def test_official_hippo_lane_builds_early_once_then_retrieves_48_48(
    tmp_path: Path,
) -> None:
    corpus = _corpus()
    build_started = threading.Event()
    release_build = threading.Event()
    calls: list[tuple[str, object]] = []

    def build(**kwargs):
        corpus_input = hippo_contract.validate_corpus_input(
            kwargs["corpus_input"]
        )
        calls.append(
            (
                "build",
                tuple(unit.text for unit in corpus_input.units),
            )
        )
        build_started.set()
        assert release_build.wait(timeout=5)
        return {"receipt_sha256": "b" * 64}

    def retrieve(**kwargs):
        query_input = hippo_contract.validate_query_input(
            kwargs["query_input"],
            expected_study_id=core.STUDY_ID,
        )
        calls.append(
            (
                "retrieve",
                tuple(
                    (row.work_id, row.query_text)
                    for row in query_input.queries
                ),
            )
        )
        return hippo_contract.RetrievalBatch(
            indices=tuple(
                (index % 5, 5, 6, 7, 8)
                for index in range(len(query_input.queries))
            ),
            receipt={"receipt_sha256": "c" * 64},
        )

    lane = _hippo_lane(
        tmp_path,
        build_callable=build,
        retrieve_callable=retrieve,
    )
    try:
        lane.start_build(corpus)
        assert build_started.wait(timeout=5)
        assert lane.build_call_count == 1
        release_build.set()

        a_items = tuple(_item(3_000 + index) for index in range(48))
        m_items = tuple(_item(4_000 + index) for index in range(48))
        a_rows = lane.retrieve(corpus, a_items)
        with pytest.raises(
            BioasqP1FormalRuntimeError,
            match="lifecycle",
        ):
            lane.retrieve(corpus, a_items)
        assert lane.retrieve_call_count == 1
        m_rows = lane.retrieve(corpus, m_items)
        assert tuple(row.work_id for row in a_rows) == tuple(
            item.work_id for item in a_items
        )
        assert tuple(row.work_id for row in m_rows) == tuple(
            item.work_id for item in m_items
        )
        assert all(row.top5_ordinals[1:] == (5, 6, 7, 8) for row in a_rows)
        assert lane.build_call_count == 1
        assert lane.retrieve_call_count == 2
        assert [name for name, _ in calls] == [
            "build",
            "retrieve",
            "retrieve",
        ]
        assert calls[0][1] == tuple(
            core.serialize_passage(row) for row in corpus.passages
        )
        assert calls[1][1] == tuple(
            (item.work_id, item.question_text) for item in a_items
        )
        assert calls[2][1] == tuple(
            (item.work_id, item.question_text) for item in m_items
        )
        assert set(lane.private_retrieval_commitments) == {
            "A_hold",
            "M_search",
        }
        for stage in ("A_hold", "M_search"):
            evidence = (
                tmp_path / "hippo" / f"{stage}.retrieval.private.json"
            )
            assert evidence.is_file()
            assert evidence.stat().st_mode & 0o777 == 0o400
            value = json.loads(evidence.read_text(encoding="ascii"))
            assert value["block"] == stage
            assert value["build_once"] is True
            assert len(value["retrieved_ordinals"]) == 48

        with pytest.raises(
            BioasqP1FormalRuntimeError,
            match="lifecycle",
        ):
            lane.retrieve(
                corpus,
                tuple(_item(5_000 + index) for index in range(48)),
            )
        assert lane.retrieve_call_count == 2
        assert [name for name, _ in calls].count("retrieve") == 2
    finally:
        release_build.set()
        lane.close()


def test_official_hippo_lane_rejects_changed_corpus_and_double_build(
    tmp_path: Path,
) -> None:
    def build(**_kwargs):
        return {"receipt_sha256": "d" * 64}

    def retrieve(**_kwargs):
        raise AssertionError("changed corpus must fail before adapter")

    corpus = _corpus()
    lane = _hippo_lane(
        tmp_path,
        build_callable=build,
        retrieve_callable=retrieve,
    )
    try:
        lane.start_build(corpus)
        with pytest.raises(
            BioasqP1FormalRuntimeError,
            match="build lifecycle",
        ):
            lane.start_build(corpus)
        with pytest.raises(
            BioasqP1FormalRuntimeError,
            match="retrieve lifecycle",
        ):
            lane.retrieve(
                _corpus(changed=True),
                tuple(_item(6_000 + index) for index in range(48)),
            )
        assert lane.build_call_count == 1
        assert lane.retrieve_call_count == 0
    finally:
        lane.close()
