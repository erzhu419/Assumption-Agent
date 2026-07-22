from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from replication_runtime.tatqa_p21_v1 import hipporag_contract as contract
from replication_runtime.tatqa_p21_v1 import hipporag_worker as worker


def _units():
    return [
        {"ordinal": 0, "text": "header", "unit_id": "T:0"},
        {"ordinal": 1, "text": "row one", "unit_id": "T:1"},
        {"ordinal": 2, "text": "row two", "unit_id": "T:2"},
        {"ordinal": 3, "text": "paragraph one", "unit_id": "P:1"},
        {"ordinal": 4, "text": "paragraph two", "unit_id": "P:2"},
        {"ordinal": 5, "text": "paragraph three", "unit_id": "P:3"},
    ]


def test_input_roundtrip_is_label_free_dynamic_and_canonical() -> None:
    payload = contract.input_payload(query="Which changed?", units=_units())
    raw = contract.canonical_json_bytes(payload)
    query, rows = contract.parse_input(raw)
    assert query == "Which changed?"
    assert len(payload["input_sha256"]) == 64
    assert tuple(row.unit_id for row in rows) == (
        "T:0",
        "T:1",
        "T:2",
        "P:1",
        "P:2",
        "P:3",
    )
    assert all("family" not in row for row in payload["units"])
    with pytest.raises(contract.TatqaP21OfficialHippoRAGError):
        contract.input_payload(
            query="Which changed?",
            units=[{**row, "answer": "x"} for row in _units()],
        )
    invalid = _units()
    invalid[-1]["unit_id"] = "P:0"
    with pytest.raises(contract.TatqaP21OfficialHippoRAGError):
        contract.input_payload(query="Which changed?", units=invalid)
    missing_header = _units()[1:]
    for ordinal, row in enumerate(missing_header):
        row["ordinal"] = ordinal
    with pytest.raises(contract.TatqaP21OfficialHippoRAGError, match="T:0"):
        contract.input_payload(query="Which changed?", units=missing_header)
    unordered = _units()
    unordered[1], unordered[2] = unordered[2], unordered[1]
    for ordinal, row in enumerate(unordered):
        row["ordinal"] = ordinal
    with pytest.raises(contract.TatqaP21OfficialHippoRAGError, match="canonical order"):
        contract.input_payload(query="Which changed?", units=unordered)


def test_serialization_keeps_duplicate_content_logically_distinct() -> None:
    units = _units()
    units[2]["text"] = units[1]["text"]
    _query, rows = contract.validate_input("q", units)
    serialized = contract.serialize_units(rows)
    assert len(set(serialized)) == len(rows)
    assert '"unit_id":"T:1"' in serialized[1]
    assert '"unit_id":"T:2"' in serialized[2]


def test_stable_top_five_uses_score_then_canonical_input_ordinal() -> None:
    _query, rows = contract.validate_input("q", _units())
    serialized = contract.serialize_units(rows)
    mapping = dict(zip(serialized, rows))
    top = contract.stable_top_k(
        retrieved_documents=reversed(serialized),
        retrieved_scores=[1.0] * len(rows),
        document_to_unit=mapping,
    )
    assert top == ("T:0", "T:1", "T:2", "P:1", "P:2")


class _Graph:
    def vcount(self):
        return 7

    def ecount(self):
        return 9


class _Core:
    def __init__(self):
        self.graph = _Graph()
        self.indexed = None
        self.requested = None

    def index(self, documents):
        self.indexed = documents

    def retrieve(self, queries, num_to_retrieve):
        self.requested = (queries, num_to_retrieve)
        return [
            SimpleNamespace(
                docs=list(reversed(self.indexed)),
                doc_scores=list(range(len(self.indexed))),
            )
        ]


def test_worker_calls_official_index_and_retrieve_once_over_all_units() -> None:
    query, rows = contract.validate_input("q", _units())
    core = _Core()
    output = worker.retrieve_with_core(core=core, query=query, units=rows)
    assert core.requested == (["q"], len(rows))
    assert len(core.indexed) == len(rows)
    assert output["top_unit_ids"] == ["T:0", "T:1", "T:2", "P:1", "P:2"]
    assert output["graph_node_count"] == 7
    assert output["graph_edge_count"] == 9
    assert contract.parse_output(contract.canonical_json_bytes(output)) == output


def test_openie_executor_is_causally_limited_to_one_worker() -> None:
    executor = worker._SingleWorkerOpenIEExecutor()
    try:
        assert executor._max_workers == 1
    finally:
        executor.shutdown(wait=True)
    with pytest.raises(
        contract.TatqaP21OfficialHippoRAGError, match="unbounded worker pool"
    ):
        worker._SingleWorkerOpenIEExecutor(max_workers=2)


def test_output_rejects_missing_or_duplicate_top_five() -> None:
    with pytest.raises(contract.TatqaP21OfficialHippoRAGError):
        contract.output_payload(
            top_unit_ids=("T:0",) * 5,
            graph_nodes=1,
            graph_edges=1,
            unit_count=6,
            input_sha256="0" * 64,
        )


def _worker_input(path: Path) -> None:
    path.write_bytes(
        contract.canonical_json_bytes(
            contract.input_payload(query="Which changed?", units=_units())
        )
    )


def _freeze_native_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in worker.NATIVE_THREAD_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key, "1")


def test_process_monitor_observes_a_real_extra_os_thread() -> None:
    monitor = worker._ProcessThreadPeakMonitor.start(os.getpid())
    release = threading.Event()
    started = [threading.Event() for _ in range(3)]

    def hold(ready: threading.Event) -> None:
        ready.set()
        release.wait(timeout=5)

    threads = [threading.Thread(target=hold, args=(ready,)) for ready in started]
    for thread in threads:
        thread.start()
    try:
        assert all(ready.wait(timeout=1) for ready in started)
        assert worker._process_thread_count(os.getpid()) >= 4
        # Give the independent monitor at least one polling interval while the
        # three extra threads are definitely alive.  A result above the frozen
        # limit proves the monitor reports procfs state instead of a hardcoded
        # value of one or two; the worker separately rejects this value.
        time.sleep(0.02)
        observed = monitor.stop()
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=1)
    assert observed >= 4


def test_worker_terminal_uses_measured_peak_and_worker_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    index_root = tmp_path / "index"
    _worker_input(input_path)
    _freeze_native_thread_environment(monkeypatch)

    class _Monitor:
        @classmethod
        def start(cls, pid):
            assert pid == os.getpid()
            return cls()

        def stop(self):
            return 2

    _query, rows = contract.parse_input(input_path.read_bytes())
    payload = contract.output_payload(
        top_unit_ids=[row.unit_id for row in rows[:5]],
        graph_nodes=7,
        graph_edges=6,
        unit_count=len(rows),
        input_sha256=contract.input_binding_sha256("Which changed?", rows),
    )
    ticks = iter((101, 205))
    monkeypatch.setattr(worker, "_ProcessThreadPeakMonitor", _Monitor)
    monkeypatch.setattr(worker, "_build_core", lambda **_kwargs: object())
    monkeypatch.setattr(worker, "retrieve_with_core", lambda **_kwargs: payload)
    monkeypatch.setattr(worker.time, "monotonic_ns", lambda: next(ticks))

    assert worker.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--index-root",
            str(index_root),
            "--llm-model",
            str(tmp_path / "llm"),
            "--embedding-model",
            str(tmp_path / "embedding"),
        ]
    ) == 0
    terminal = json.loads(capsys.readouterr().out)
    assert terminal["model_execution_started_monotonic_ns"] == 101
    assert terminal["model_execution_finished_monotonic_ns"] == 205
    assert terminal["observed_process_thread_peak"] == 2
    assert terminal["configured_torch_intraop_threads"] == 1
    assert terminal["configured_torch_interop_threads"] == 1
    assert output_path.is_file()


def test_worker_fails_closed_before_output_when_measured_peak_exceeds_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    _worker_input(input_path)
    _freeze_native_thread_environment(monkeypatch)

    class _Monitor:
        @classmethod
        def start(cls, _pid):
            return cls()

        def stop(self):
            return 3

    _query, rows = contract.parse_input(input_path.read_bytes())
    payload = contract.output_payload(
        top_unit_ids=[row.unit_id for row in rows[:5]],
        graph_nodes=7,
        graph_edges=6,
        unit_count=len(rows),
        input_sha256=contract.input_binding_sha256("Which changed?", rows),
    )
    ticks = iter((101, 205))
    monkeypatch.setattr(worker, "_ProcessThreadPeakMonitor", _Monitor)
    monkeypatch.setattr(worker, "_build_core", lambda **_kwargs: object())
    monkeypatch.setattr(worker, "retrieve_with_core", lambda **_kwargs: payload)
    monkeypatch.setattr(worker.time, "monotonic_ns", lambda: next(ticks))

    with pytest.raises(
        contract.TatqaP21OfficialHippoRAGError, match="thread peak"
    ):
        worker.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--index-root",
                str(tmp_path / "index"),
                "--llm-model",
                str(tmp_path / "llm"),
                "--embedding-model",
                str(tmp_path / "embedding"),
            ]
        )
    assert not output_path.exists()


def test_worker_rejects_unbounded_native_thread_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_native_thread_environment(monkeypatch)
    worker._require_native_thread_environment()
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    with pytest.raises(
        contract.TatqaP21OfficialHippoRAGError, match="BLAS/OpenMP"
    ):
        worker._require_native_thread_environment()
