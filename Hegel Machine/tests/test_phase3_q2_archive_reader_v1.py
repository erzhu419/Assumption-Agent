from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hegel_machine import phase3_q1_formal_archive_contract_v1 as formal
from hegel_machine import phase3_q1_quotient_contract_v1 as quotient
from hegel_machine import phase3_q1_universe_v1 as universe
from hegel_machine import phase3_q2_admission_v1 as admission
from hegel_machine import phase3_q2_archive_reader_v1 as reader


Q1_V8C_CONFIG_PATH = ROOT / "config/phase3_q1_completion_successor_v8c.json"


def _root(value: int) -> bytes:
    return bytes((value,)) * 32


def _class_records(
    input_signature_id: int,
    cells_by_class: tuple[tuple[formal.Q1BehaviorCellV1, ...], ...],
    *,
    output_sort_id: quotient.OutputSortId = quotient.OutputSortId.BIT,
) -> tuple[formal.Q1QuotientClassRecordV1, ...]:
    universe_root = universe.production_universe_v1(input_signature_id).universe_root
    behaviors = tuple(
        formal.Q1BehaviorBlobV1(
            input_signature_id,
            universe_root,
            output_sort_id,
            cells,
        )
        for cells in cells_by_class
    )
    ordered = tuple(sorted(behaviors, key=lambda item: (item.behavior_id, item.canonical_bytes)))
    return tuple(
        formal.Q1QuotientClassRecordV1(
            input_signature_id,
            universe_root,
            index,
            behavior,
            index,
            1,
            _root(10 + index),
            1,
            1,
            1,
            _root(20 + index),
            1,
        )
        for index, behavior in enumerate(ordered)
    )


def _actual_inputs() -> tuple[bytes, bytes, Path]:
    config = json.loads(Q1_V8C_CONFIG_PATH.read_text(encoding="utf-8"))
    fixed = config["fixed_paths"]
    terminal_path = Path(fixed["terminal_path"])
    artifact_path = Path(fixed["artifact_path"])
    endpoint_root = Path(config["read_only_replay"]["python_output_root"])
    if not terminal_path.is_file() or not artifact_path.is_file() or not endpoint_root.is_dir():
        pytest.skip("the external sealed Q1 v8c inputs are not present")
    return terminal_path.read_bytes(), artifact_path.read_bytes(), endpoint_root


def test_typed_class_decoder_replays_canonical_q1_records() -> None:
    rows = len(universe.production_universe_v1(2).rows)
    records = _class_records(
        2,
        (
            tuple(formal.Q1BehaviorCellV1.exact(0) for _ in range(rows)),
            tuple(formal.Q1BehaviorCellV1.exact(1) for _ in range(rows)),
        ),
    )
    framed = b"".join(
        formal.frame_canonical_record_v1(record.canonical_object())
        for record in records
    )
    assert reader.decode_q1_class_records_v1(
        framed,
        input_signature_id=2,
        universe_root=records[0].universe_root,
    ) == records
    with pytest.raises(reader.Q2ArchiveReaderError, match="truncated frame"):
        reader.decode_q1_class_records_v1(
            framed + b"\x00",
            input_signature_id=2,
            universe_root=records[0].universe_root,
        )


def test_actual_sealed_python_class_streams_replay_in_place() -> None:
    terminal, artifact, endpoint_root = _actual_inputs()
    admitted = admission.admit_q1_completion_v1(terminal, artifact)
    odd, sink = reader.read_q1_class_streams_v1(
        endpoint_root,
        admitted,
    )
    assert (len(odd.records), odd.framed_bytes) == (418, 852_581)
    assert (len(sink.records), sink.framed_bytes) == (89, 55_084)
    assert odd.framed_relative_paths == (
        "signature-01/stream-03/chunk-000000.framed",
    )
    assert sink.framed_relative_paths == (
        "signature-02/stream-03/chunk-000000.framed",
    )
    assert odd.endpoint_root == sink.endpoint_root == endpoint_root.resolve()
