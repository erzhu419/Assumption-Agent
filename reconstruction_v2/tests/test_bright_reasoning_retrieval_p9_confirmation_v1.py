from __future__ import annotations

from collections import Counter

import pytest

from assumption_agent.benchmarks import bright_reasoning_retrieval_acquisition_v1 as source
from assumption_agent.benchmarks import bright_reasoning_retrieval_p9_confirmation_v1 as confirmation


def _rows(counts: dict[str, int]) -> tuple[source.SourceItem, ...]:
    rows: list[source.SourceItem] = []
    for family in source.FAMILY_ORDER:
        for index in range(counts[family]):
            rows.append(
                source.SourceItem(
                    family=family,
                    source_id=f"{family}-{index}",
                    query=f"query {family} {index}",
                    excluded_ids=(),
                    gold_ids=(f"gold-{family}-{index}",),
                )
            )
    return tuple(rows)


def test_confirmation_selection_is_frozen_family_slice() -> None:
    rows = _rows({"BIOLOGY": 28, "ECONOMICS": 28, "ROBOTICS": 26})
    selected = confirmation.select_confirmation_rows(rows)
    assert len(selected) == 33
    assert Counter(row.family for row in selected) == Counter(
        {family: 11 for family in source.FAMILY_ORDER}
    )
    for family in source.FAMILY_ORDER:
        family_ids = [row.source_id for row in selected if row.family == family]
        assert family_ids == [f"{family}-{index}" for index in range(15, 26)]


def test_confirmation_selection_rejects_insufficient_family() -> None:
    rows = _rows({"BIOLOGY": 28, "ECONOMICS": 28, "ROBOTICS": 25})
    with pytest.raises(confirmation.BrightP9ConfirmationError):
        confirmation.select_confirmation_rows(rows)


def test_confirmation_decision_requires_every_family_against_both() -> None:
    aggregates = {
        "P9": {
            "family_sum_integer_ndcg": {
                "BIOLOGY": 11,
                "ECONOMICS": 11,
                "ROBOTICS": 11,
            },
            "sum_integer_ndcg": 33,
        },
        "RAW": {
            "family_sum_integer_ndcg": {
                "BIOLOGY": 10,
                "ECONOMICS": 10,
                "ROBOTICS": 10,
            },
            "sum_integer_ndcg": 30,
        },
        "HippoRAG": {
            "family_sum_integer_ndcg": {
                "BIOLOGY": 10,
                "ECONOMICS": 10,
                "ROBOTICS": 10,
            },
            "sum_integer_ndcg": 30,
        },
    }
    passed, raw_delta, hippo_delta = confirmation.confirmation_passed(aggregates)
    assert passed is True
    assert set(raw_delta.values()) == {1}
    assert set(hippo_delta.values()) == {1}
    aggregates["HippoRAG"]["family_sum_integer_ndcg"]["ROBOTICS"] = 11
    passed, _raw_delta, _hippo_delta = confirmation.confirmation_passed(aggregates)
    assert passed is False
