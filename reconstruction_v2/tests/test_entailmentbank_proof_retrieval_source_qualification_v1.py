from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from assumption_agent.benchmarks import (
    entailmentbank_proof_retrieval_source_qualification_v1 as qualifier,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _row(
    item_id: str,
    leaf_count: int,
    *,
    question: str | None = None,
    hypothesis: str | None = None,
) -> dict[str, Any]:
    triples = {
        f"sent{index}": f"private fact {item_id} {index}"
        for index in range(1, 9)
    }
    if leaf_count == 2:
        proof = "sent1 & sent2 -> hypothesis;"
        intermediates: dict[str, str] = {}
    elif leaf_count == 3:
        proof = "sent1 & sent2 -> int1: private intermediate; int1 & sent3 -> hypothesis;"
        intermediates = {"int1": "private intermediate"}
    elif leaf_count == 4:
        proof = (
            "sent1 & sent2 -> int1: private intermediate one; "
            "sent3 & sent4 -> int2: private intermediate two; "
            "int1 & int2 -> hypothesis;"
        )
        intermediates = {
            "int1": "private intermediate one",
            "int2": "private intermediate two",
        }
    else:
        raise AssertionError(leaf_count)
    return {
        "id": item_id,
        "question": question or f"private question {item_id}",
        "answer": f"private answer {item_id}",
        "hypothesis": hypothesis or f"private hypothesis {item_id}",
        "context": "private context",
        "proof": proof,
        "meta": {
            "triples": triples,
            "intermediate_conclusions": intermediates,
            "distractors": [
                f"sent{index}" for index in range(leaf_count + 1, 9)
            ],
        },
    }


def _three(prefix: str) -> list[dict[str, Any]]:
    return [
        _row(f"{prefix}-two", 2),
        _row(f"{prefix}-three", 3),
        _row(f"{prefix}-four", 4),
    ]


def _demands(value: int) -> dict[str, int]:
    return {family: value for family in qualifier.FAMILY_ORDER}


def _verify_self_hash(receipt: dict[str, Any]) -> None:
    body = copy.deepcopy(receipt)
    declared = body.pop("qualification_sha256")
    assert hashlib.sha256(_canonical(body)).hexdigest() == declared


def test_balanced_proof_families_pass_without_selection_or_text_output() -> None:
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(_three("train")),
        _jsonl(_three("dev")),
        source_binding={"fixture": "synthetic"},
        train_demands=_demands(1),
        dev_demands=_demands(1),
    )
    assert receipt["status"] == "qualified_source_capacity_no_selection"
    assert receipt["terminal_reason_counts"] == {
        "unsatisfied_capacity_count": 0
    }
    for split in qualifier.SPLIT_ORDER:
        capacity = receipt["simultaneous_component_disjoint_capacity"][split]
        assert capacity["maximum_flow_assigned_total"] == 3
        assert capacity["simultaneous_family_capacity_saturated"] is True
    serialized = _canonical(receipt).decode("ascii")
    for forbidden in (
        "private question",
        "private answer",
        "private hypothesis",
        "private fact",
        "private intermediate",
    ):
        assert forbidden not in serialized
    assert receipt["claim_boundary"]["test_payload_hashed_opened_or_parsed"] is False
    _verify_self_hash(receipt)


def test_reader_incompatible_and_formally_ineligible_rows_are_excluded() -> None:
    train = _three("train")
    invalid = _row("invalid", 2)
    invalid["meta"]["triples"] = []
    too_short = _row("too-short", 2)
    too_short["meta"]["triples"] = dict(
        list(too_short["meta"]["triples"].items())[:7]
    )
    too_short["meta"]["distractors"] = ["sent3", "sent4", "sent5", "sent6", "sent7"]
    train.extend((invalid, too_short))
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(train),
        _jsonl(_three("dev")),
        source_binding={},
        train_demands=_demands(1),
        dev_demands=_demands(1),
    )
    assert receipt["status"] == "qualified_source_capacity_no_selection"
    aggregate = receipt["split_aggregates"]["train"]
    assert aggregate["reader_incompatible_reason_counts"] == {"triples": 1}
    assert aggregate["formal_ineligible_reason_counts"] == {
        "formal_context_size": 1
    }


def test_cross_split_normalized_question_component_is_excluded() -> None:
    train = _three("train")
    dev = _three("dev")
    train[0]["question"] = "  Shared   Question  "
    dev[0]["question"] = "shared question"
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(train),
        _jsonl(dev),
        source_binding={},
        train_demands=_demands(1),
        dev_demands=_demands(1),
    )
    assert receipt["status"] == "terminal_source_infeasible_no_selection"
    graph = receipt["candidate_and_component_aggregates"]["component_graph"]
    assert graph["cross_split_component_count"] == 1
    for split in qualifier.SPLIT_ORDER:
        counts = receipt["candidate_and_component_aggregates"][
            "candidate_splits"
        ][split]["cross_split_component_excluded_candidate_counts"]
        assert counts["TWO_LEAF"] == 1


def test_documentation_example_component_is_excluded() -> None:
    train = _three("train")
    train.append(_row(qualifier.DOCUMENTATION_EXAMPLE_ID, 2))
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(train),
        _jsonl(_three("dev")),
        source_binding={},
        train_demands=_demands(1),
        dev_demands=_demands(1),
    )
    assert receipt["status"] == "qualified_source_capacity_no_selection"
    graph = receipt["candidate_and_component_aggregates"]["component_graph"]
    assert graph["documentation_example_component_count"] == 1
    counts = receipt["candidate_and_component_aggregates"]["candidate_splits"][
        "train"
    ]["documentation_example_component_excluded_candidate_counts"]
    assert counts["TWO_LEAF"] == 1


def test_unknown_proof_reference_is_reader_incompatible_not_a_gold_leaf() -> None:
    train = _three("train")
    unknown = _row("unknown", 2)
    unknown["proof"] = "sent1 & missing -> hypothesis;"
    train.append(unknown)
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(train),
        _jsonl(_three("dev")),
        source_binding={},
        train_demands=_demands(1),
        dev_demands=_demands(1),
    )
    assert receipt["status"] == "qualified_source_capacity_no_selection"
    assert receipt["split_aggregates"]["train"][
        "reader_incompatible_reason_counts"
    ] == {"proof_LHS_unknown_reference": 1}


def test_duplicate_item_or_question_component_has_unit_capacity() -> None:
    train = _three("train")
    duplicate_question = _row("duplicate-question", 2)
    duplicate_question["question"] = train[0]["question"]
    train.append(duplicate_question)
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(train),
        _jsonl(_three("dev")),
        source_binding={},
        train_demands={
            "TWO_LEAF": 2,
            "THREE_LEAF": 1,
            "FOUR_FIVE_LEAF": 1,
        },
        dev_demands=_demands(1),
    )
    assert receipt["status"] == "terminal_source_infeasible_no_selection"
    graph = receipt["candidate_and_component_aggregates"]["component_graph"]
    assert graph["multi_row_component_count"] == 1
    capacity = receipt["simultaneous_component_disjoint_capacity"]["train"]
    assert capacity["assignable_component_counts"]["TWO_LEAF"] == 1


def test_duplicate_JSON_keys_are_aggregated_without_content() -> None:
    receipt = qualifier.qualify_decoded_sources(
        _jsonl(_three("train")) + b'{"id":"x","id":"y"}\n',
        _jsonl(_three("dev")),
        source_binding={},
        train_demands=_demands(1),
        dev_demands=_demands(1),
    )
    assert receipt["status"] == "qualified_source_capacity_no_selection"
    assert receipt["split_aggregates"]["train"][
        "reader_incompatible_reason_counts"
    ] == {"json_line": 1}
