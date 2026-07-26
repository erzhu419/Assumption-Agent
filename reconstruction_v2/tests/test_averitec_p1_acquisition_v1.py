from __future__ import annotations

import json

from assumption_agent.benchmarks.averitec_p1_acquisition_v1 import (
    A_FORM,
    A_HOLD,
    BLOCK_ORDER,
    CAUSAL,
    F_SEARCH,
    M_SEARCH,
    NUMERICAL,
    QUOTE,
    acquire_from_rows,
)


def _row(index: int, family: str, split: str) -> dict[str, object]:
    type_value = {
        CAUSAL: "Causal Claim",
        QUOTE: "Quote Verification",
        NUMERICAL: "Numerical Claim",
    }[family]
    return {
        "claim": f"PRIVATE {split} {family} CLAIM {index}",
        "claim_types": [type_value],
        "questions": [
            {
                "question": f"PRIVATE QUESTION {split} {family} {index} A",
                "answers": [
                    {
                        "answer": f"PRIVATE ANSWER {split} {family} {index} A",
                        "answer_type": "Extractive",
                    }
                ],
            },
            {
                "question": f"PRIVATE QUESTION {split} {family} {index} B",
                "answers": [
                    {
                        "answer": f"PRIVATE ANSWER {split} {family} {index} B",
                        "answer_type": "Abstractive",
                    }
                ],
            },
        ],
        "label": "Supported",
        "justification": "PRIVATE JUSTIFICATION",
    }


def test_acquisition_builds_component_disjoint_views_and_late_qrels() -> None:
    train = [
        _row(index, family, "train")
        for family in (CAUSAL, QUOTE, NUMERICAL)
        for index in range(4)
    ]
    dev = [
        _row(index, family, "dev")
        for family in (CAUSAL, QUOTE, NUMERICAL)
        for index in range(4)
    ]
    quotas = {
        A_FORM: {family: 2 for family in (CAUSAL, QUOTE, NUMERICAL)},
        F_SEARCH: {family: 1 for family in (CAUSAL, QUOTE, NUMERICAL)},
        A_HOLD: {family: 2 for family in (CAUSAL, QUOTE, NUMERICAL)},
        M_SEARCH: {family: 1 for family in (CAUSAL, QUOTE, NUMERICAL)},
    }
    payloads, aggregate = acquire_from_rows(
        train_rows=train,
        dev_rows=dev,
        secret=b"x" * 32,
        block_quotas=quotas,
    )
    assert tuple(quotas) == BLOCK_ORDER
    assert "F_search.qrels" not in payloads
    assert {
        "A_form.qrels",
        "A_hold.qrels",
        "M_search.qrels",
    } <= set(payloads)
    all_item_ids = []
    for block in BLOCK_ORDER:
        view = payloads[f"{block}.view"]
        assert len(view["queries"]) == sum(quotas[block].values())
        assert len(view["corpus"]) == 2 * len(view["queries"])
        all_item_ids.extend(row["item_id"] for row in view["queries"])
        if block != F_SEARCH:
            qrels = payloads[f"{block}.qrels"]
            assert all(
                len(row["qrel_document_ordinals"]) == 2
                for row in qrels["rows"]
            )
    assert len(all_item_ids) == len(set(all_item_ids))
    assert aggregate["selected_component_count"] == len(all_item_ids)
    rendered = json.dumps(payloads, sort_keys=True)
    assert "PRIVATE JUSTIFICATION" not in rendered
    assert '"label"' not in rendered


def test_cross_split_collision_component_is_excluded() -> None:
    shared_train = _row(999, CAUSAL, "shared")
    shared_dev = json.loads(json.dumps(shared_train))
    train = [shared_train] + [
        _row(index, family, "train")
        for family in (CAUSAL, QUOTE, NUMERICAL)
        for index in range(4)
    ]
    dev = [shared_dev] + [
        _row(index, family, "dev")
        for family in (CAUSAL, QUOTE, NUMERICAL)
        for index in range(4)
    ]
    quotas = {
        A_FORM: {family: 2 for family in (CAUSAL, QUOTE, NUMERICAL)},
        F_SEARCH: {family: 1 for family in (CAUSAL, QUOTE, NUMERICAL)},
        A_HOLD: {family: 2 for family in (CAUSAL, QUOTE, NUMERICAL)},
        M_SEARCH: {family: 1 for family in (CAUSAL, QUOTE, NUMERICAL)},
    }
    _payloads, aggregate = acquire_from_rows(
        train_rows=train,
        dev_rows=dev,
        secret=b"y" * 32,
        block_quotas=quotas,
    )
    assert aggregate["qualification"]["cross_split_component_count"] == 1
    assert (
        aggregate["qualification"]["cross_split_row_exclusion_count"] == 2
    )
