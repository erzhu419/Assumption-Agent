from __future__ import annotations

import csv
import io

import pytest

from assumption_agent.gscl_arn_raw_adapter_v1 import (
    ArnRawAdapterError,
    ArnTopology,
    parse_arn_csv_bytes,
)
from assumption_agent.benchmarks.gscl_arn_intrinsic_protocol_v1 import (
    OFFICIAL_HEADER,
)


def _csv(rows: list[list[str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, dialect="excel", lineterminator="\r\n")
    writer.writerow(OFFICIAL_HEADER)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _rows() -> list[list[str]]:
    return [
        [
            "1",
            "Proverb one",
            "A quiet query.",
            "The first path.",
            "The second path.",
            "High",
            "Far",
            "A",
        ],
        [
            "3",
            "Proverb two",
            "Another query.",
            "Alpha route.",
            "Beta route.",
            "low",
            "near",
            "Beta route.",
        ],
    ]


def test_source_free_parser_maps_only_canonical_fields() -> None:
    rows = parse_arn_csv_bytes(
        _csv(_rows()),
        expected_topology=ArnTopology(
            row_count=2,
            id_minimum=1,
            id_maximum=3,
            missing_ids=(2,),
            cell_counts={
                "far_high": 1,
                "far_low": 0,
                "near_high": 0,
                "near_low": 1,
            },
        ),
    )
    assert [row.source_id for row in rows] == ["1", "3"]
    assert [row.gold_choice for row in rows] == [
        "first_choice",
        "second_choice",
    ]
    assert rows[0].analogy_level == "far"
    assert rows[0].distractor_similarity == "high"


@pytest.mark.parametrize(
    "mutator,issue",
    [
        (
            lambda rows: rows
            + [
                [
                    "1",
                    "P",
                    "Q",
                    "A",
                    "B",
                    "low",
                    "near",
                    "A",
                ]
            ],
            "duplicated",
        ),
        (
            lambda rows: [
                [*rows[0][:-1], "0"],
                rows[1],
            ],
            "unsupported",
        ),
        (
            lambda rows: [
                [
                    *rows[0][:3],
                    rows[0][3],
                    rows[0][3],
                    *rows[0][5:],
                ],
                rows[1],
            ],
            "identical",
        ),
        (
            lambda rows: [
                [*rows[0][:5], "medium", *rows[0][6:]],
                rows[1],
            ],
            "canonical",
        ),
    ],
)
def test_parser_fails_closed(
    mutator, issue: str
) -> None:
    with pytest.raises(ArnRawAdapterError, match=issue):
        parse_arn_csv_bytes(
            _csv(mutator(_rows())),
            expected_topology=None,
        )


def test_header_width_topology_and_utf8_fail_closed() -> None:
    raw = _csv(_rows())
    with pytest.raises(ArnRawAdapterError, match="header"):
        parse_arn_csv_bytes(
            raw.replace(b"correct_answer", b"answer", 1),
            expected_topology=None,
        )
    with pytest.raises(ArnRawAdapterError, match="width"):
        parse_arn_csv_bytes(
            raw + b"4,too,few\r\n",
            expected_topology=None,
        )
    with pytest.raises(ArnRawAdapterError, match="row count"):
        parse_arn_csv_bytes(
            raw,
            expected_topology=ArnTopology(
                row_count=3,
                id_minimum=1,
                id_maximum=3,
                missing_ids=(2,),
                cell_counts={
                    "far_high": 1,
                    "far_low": 0,
                    "near_high": 0,
                    "near_low": 1,
                },
            ),
        )
    with pytest.raises(ArnRawAdapterError, match="UTF-8"):
        parse_arn_csv_bytes(
            b",".join(name.encode() for name in OFFICIAL_HEADER)
            + b"\r\n1,\xff",
            expected_topology=None,
        )


def test_parser_does_not_accept_zero_based_or_fuzzy_answer() -> None:
    for answer in ("0", "probably first", "choice"):
        rows = _rows()
        rows[0][-1] = answer
        with pytest.raises(
            ArnRawAdapterError, match="unsupported"
        ):
            parse_arn_csv_bytes(
                _csv(rows), expected_topology=None
            )
