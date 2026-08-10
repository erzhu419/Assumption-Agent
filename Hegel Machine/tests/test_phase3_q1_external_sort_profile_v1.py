from __future__ import annotations

import importlib
from pathlib import Path
import random
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

profile = importlib.import_module("hegel_machine.phase3_q1_external_sort_profile_v1")
contract = importlib.import_module("hegel_machine.phase3_q1_formal_archive_contract_v1")


def test_run_header_is_bit_exact_68_bytes() -> None:
    payload = b"abc"
    header = profile.run_header_v1(
        input_signature_id=1,
        stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
        level=2,
        run_index=3,
        record_count=4,
        payload=payload,
    )
    assert len(header) == contract.EXTERNAL_SORT_RUN_HEADER_BYTES == 68
    assert header[:8] == b"HGQ1RUN1"
    assert int.from_bytes(header[8:10], "big") == 1
    assert int.from_bytes(header[10:12], "big") == 1
    assert int.from_bytes(header[12:14], "big") == 1
    assert int.from_bytes(header[14:16], "big") == 2
    assert int.from_bytes(header[16:20], "big") == 3
    assert int.from_bytes(header[20:28], "big") == 4
    assert int.from_bytes(header[28:36], "big") == len(payload)


def test_run_file_replay_rejects_header_and_payload_tamper() -> None:
    rows = ((b"a", b"one"), (b"b", b"two"))
    payload = b"".join(profile.external_sort_row_bytes_v1(*row) for row in rows)
    header = profile.run_header_v1(
        input_signature_id=1,
        stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
        level=0,
        run_index=0,
        record_count=2,
        payload=payload,
    )
    run_file = header + payload
    assert profile.replay_run_file_v1(
        run_file,
        input_signature_id=1,
        stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
        level=0,
        run_index=0,
    ) == rows
    for tampered in (
        bytes((run_file[0] ^ 1,)) + run_file[1:],
        run_file[:-1] + bytes((run_file[-1] ^ 1,)),
    ):
        with pytest.raises(profile.Q1ExternalSortError) as caught:
            profile.replay_run_file_v1(
                tampered,
                input_signature_id=1,
                stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
                level=0,
                run_index=0,
            )
        assert caught.value.code == "REJECT_Q1_SORT_RUN_REPLAY"


@pytest.mark.parametrize(
    ("initial", "expected"),
    (
        (1, (1,)),
        (16, (16, 1)),
        (17, (17, 2, 1)),
        (256, (256, 16, 1)),
        (257, (257, 17, 2, 1)),
    ),
)
def test_merge_shape_is_frozen_for_fan_in_boundaries(initial, expected) -> None:
    assert profile.external_sort_merge_shape_v1(initial) == expected


def test_projection_is_shuffle_invariant_and_charges_physical_reserve() -> None:
    rows = tuple(
        (
            index.to_bytes(4, "big"),
            (b"record-" + index.to_bytes(4, "big")) * (1 + index % 3),
        )
        for index in range(100)
    )
    shuffled = list(rows)
    random.Random(7319).shuffle(shuffled)
    left = profile.project_external_sort_v1(
        rows,
        input_signature_id=1,
        stream_kind_id=contract.ArchiveStreamKindId.COHORT,
    )
    right = profile.project_external_sort_v1(
        shuffled,
        input_signature_id=1,
        stream_kind_id=contract.ArchiveStreamKindId.COHORT,
    )
    assert left == right
    assert left.initial_run_count == 1
    assert left.merge_level_count == 0
    assert left.final_run_bytes == contract.EXTERNAL_SORT_RUN_HEADER_BYTES + left.input_payload_bytes
    assert left.charged_scratch_high_water_bytes > left.logical_scratch_high_water_bytes
    assert len(left.diagnostic_root) == 32


def test_projection_rejects_duplicate_key_and_bool_signature_alias() -> None:
    with pytest.raises(profile.Q1ExternalSortError) as error:
        profile.project_external_sort_v1(
            ((b"a", b"one"), (b"a", b"two")),
            input_signature_id=1,
            stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
        )
    assert error.value.code == "REJECT_Q1_SORT_INPUT"
    with pytest.raises(profile.Q1ExternalSortError):
        profile.project_external_sort_v1(
            ((b"a", b"one"),),
            input_signature_id=True,
            stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
        )
    with pytest.raises(profile.Q1ExternalSortError) as caught:
        profile.ScratchEventV1(True, 1, b"run", 0, 68, 68, 8192).canonical_object()
    assert caught.value.code == "REJECT_Q1_SCRATCH_EVENT"
    with pytest.raises(profile.Q1ExternalSortError) as caught:
        profile.ScratchEventV1(0, True, b"run", 0, 68, 68, 8192).canonical_object()
    assert caught.value.code == "REJECT_Q1_SCRATCH_EVENT"


def test_projection_replays_every_created_run_before_free(monkeypatch) -> None:
    monkeypatch.setattr(profile, "EXTERNAL_SORT_RUN_PAYLOAD_LIMIT_BYTES", 32)
    rows = tuple((bytes((index,)), b"r" * 8) for index in range(20))
    projection = profile.project_external_sort_v1(
        rows,
        input_signature_id=1,
        stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
    )
    assert projection.initial_run_count > 1
    assert projection.merge_level_count > 0
    assert projection.scratch_event_count == 4 * sum(
        profile.external_sort_merge_shape_v1(projection.initial_run_count)
    )
