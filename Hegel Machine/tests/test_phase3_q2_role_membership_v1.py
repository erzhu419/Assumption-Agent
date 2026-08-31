from __future__ import annotations

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
from hegel_machine import phase3_q2_role_membership_v1 as membership
from hegel_machine.phase3_q1_formal_archive_contract_v1 import Q1_OUTPUT_SLOT_NAMES


def _root(value: int) -> bytes:
    return bytes((value,)) * 32


def _class_record(
    input_signature_id: int,
    cells: tuple[formal.Q1BehaviorCellV1, ...],
    *,
    output_sort_id: quotient.OutputSortId,
) -> formal.Q1QuotientClassRecordV1:
    universe_root = universe.production_universe_v1(input_signature_id).universe_root
    behavior = formal.Q1BehaviorBlobV1(
        input_signature_id,
        universe_root,
        output_sort_id,
        cells,
    )
    return formal.Q1QuotientClassRecordV1(
        input_signature_id,
        universe_root,
        0,
        behavior,
        0,
        1,
        _root(1),
        1,
        1,
        1,
        _root(2),
        1,
    )


def _stream(record: formal.Q1QuotientClassRecordV1) -> reader.Q1ClassStreamV1:
    return reader.Q1ClassStreamV1(
        Path("/sealed/q1-python-output"),
        "python",
        record.input_signature_id,
        record.universe_root,
        _root(3),
        _root(4),
        _root(5),
        (record,),
        ("class.framed",),
        len(formal.frame_canonical_record_v1(record.canonical_object())),
    )


def test_q1_cell_bridge_preserves_exact_bool_bit_and_bottom_wires() -> None:
    assert membership.q1_cell_matches_target_bit_v1(
        quotient.OutputSortId.BOOL,
        formal.Q1BehaviorCellV1.exact(True),
        1,
    )
    assert not membership.q1_cell_matches_target_bit_v1(
        quotient.OutputSortId.BOOL,
        formal.Q1BehaviorCellV1.exact(False),
        1,
    )
    assert membership.q1_cell_matches_target_bit_v1(
        quotient.OutputSortId.BIT,
        formal.Q1BehaviorCellV1.exact(0),
        0,
    )
    assert not membership.q1_cell_matches_target_bit_v1(
        quotient.OutputSortId.BIT,
        formal.Q1BehaviorCellV1.bottom(),
        0,
    )
    with pytest.raises(ValueError):
        membership.q1_cell_matches_target_bit_v1(
            quotient.OutputSortId.BIT,
            formal.Q1BehaviorCellV1.exact(True),
            1,
        )


def test_full_row_fold_rejects_one_mismatch_or_bottom() -> None:
    row_count = len(universe.production_universe_v1(2).rows)
    target = (1,) * row_count
    exact = _class_record(
        2,
        tuple(formal.Q1BehaviorCellV1.exact(1) for _ in range(row_count)),
        output_sort_id=quotient.OutputSortId.BIT,
    )
    assert membership.q1_class_matches_truth_v1(exact, target)
    changed = list(exact.behavior.cells)
    changed[-1] = formal.Q1BehaviorCellV1.exact(0)
    mismatch = _class_record(
        2,
        tuple(changed),
        output_sort_id=quotient.OutputSortId.BIT,
    )
    assert not membership.q1_class_matches_truth_v1(mismatch, target)
    changed[-1] = formal.Q1BehaviorCellV1.bottom()
    bottom = _class_record(
        2,
        tuple(changed),
        output_sort_id=quotient.OutputSortId.BIT,
    )
    assert not membership.q1_class_matches_truth_v1(bottom, target)


def test_preregistered_odd_zero_and_sink_positive_project_to_pass() -> None:
    odd_commitment, sink_commitment = admission.Q2_ROLE_COMMITMENTS
    odd_rows = odd_commitment.row_count
    sink_rows = sink_commitment.row_count
    odd_record = _class_record(
        1,
        tuple(formal.Q1BehaviorCellV1.exact(0) for _ in range(odd_rows)),
        output_sort_id=quotient.OutputSortId.BIT,
    )
    sink_record = _class_record(
        2,
        tuple(formal.Q1BehaviorCellV1.exact(1) for _ in range(sink_rows)),
        output_sort_id=quotient.OutputSortId.BIT,
    )
    odd = membership.evaluate_role_membership_v1(
        _stream(odd_record),
        odd_commitment,
        (1,) + (0,) * (odd_rows - 1),
    )
    sink = membership.evaluate_role_membership_v1(
        _stream(sink_record),
        sink_commitment,
        (1,) * sink_rows,
    )
    assert odd.match_count == 0
    assert odd.membership_verdict == membership.OUTSIDE_CLOSURE
    assert sink.match_count == 1
    assert sink.membership_verdict == membership.IN_CLOSURE
    assert membership.combine_q2_role_results_v1(odd, sink) == membership.PASS_STATUS
    slots = tuple(
        (index, name, bytes.fromhex(root[7:]))
        for index, (name, root) in enumerate(
            zip(
                Q1_OUTPUT_SLOT_NAMES,
                admission.Q1_V8C_BINDING.ordered_q1_output_slots,
                strict=True,
            ),
            1,
        )
    )
    predecessor = admission.Q1CompletionAdmissionV1(
        admission.Q1_V8C_BINDING,
        slots,
    )
    artifact = membership.build_q2_membership_artifact_v1(
        _root(9),
        _root(10),
        _root(11),
        predecessor,
        odd,
        sink,
        observed_wall_time_ns=1,
        observed_peak_rss_bytes=1,
    )
    assert membership.decode_q2_membership_artifact_v1(
        artifact.canonical_bytes
    ) == artifact
    assert artifact.status == membership.PASS_STATUS
    changed = bytearray(artifact.canonical_bytes)
    changed[-1] ^= 1
    with pytest.raises(membership.Q2RoleMembershipError):
        membership.decode_q2_membership_artifact_v1(bytes(changed))


def test_complete_but_wrong_scientific_disposition_is_not_retryable_failure() -> None:
    odd_commitment, sink_commitment = admission.Q2_ROLE_COMMITMENTS
    odd_record = _class_record(
        1,
        tuple(
            formal.Q1BehaviorCellV1.exact(0)
            for _ in range(odd_commitment.row_count)
        ),
        output_sort_id=quotient.OutputSortId.BIT,
    )
    sink_record = _class_record(
        2,
        tuple(
            formal.Q1BehaviorCellV1.exact(0)
            for _ in range(sink_commitment.row_count)
        ),
        output_sort_id=quotient.OutputSortId.BIT,
    )
    odd = membership.evaluate_role_membership_v1(
        _stream(odd_record),
        odd_commitment,
        (0,) * odd_commitment.row_count,
    )
    sink = membership.evaluate_role_membership_v1(
        _stream(sink_record),
        sink_commitment,
        (1,) * sink_commitment.row_count,
    )
    assert membership.combine_q2_role_results_v1(odd, sink) == (
        membership.COMPLETE_FAIL_STATUS
    )
