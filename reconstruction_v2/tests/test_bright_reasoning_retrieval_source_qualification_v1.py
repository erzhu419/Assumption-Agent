from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    bright_reasoning_retrieval_source_qualification_v1 as qualification,
)


def _documents(prefix: str, count: int = 4) -> list[dict[str, Any]]:
    return [{"id": f"{prefix}-doc-{index}"} for index in range(count)]


def _examples(prefix: str, count: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "excluded_ids": [],
            "gold_ids": [f"{prefix}-doc-{index}"],
            "id": f"{prefix}-item-{index}",
            "query": f"private query {prefix} {index}",
        }
        for index in range(count)
    ]


def _family_rows(count: int = 2) -> tuple[dict[str, Any], dict[str, Any]]:
    documents = {
        family: _documents(family.lower())
        for family in qualification.FAMILY_ORDER
    }
    examples = {
        family: _examples(family.lower(), count=count)
        for family in qualification.FAMILY_ORDER
    }
    return documents, examples


def _demands(value: int) -> dict[str, int]:
    return {family: value for family in qualification.FAMILY_ORDER}


def _verify_hash(receipt: dict[str, Any]) -> None:
    body = copy.deepcopy(receipt)
    declared = body.pop("qualification_sha256")
    assert hashlib.sha256(qualification.canonical_json(body)).hexdigest() == declared


def test_balanced_components_pass_without_private_output() -> None:
    documents, examples = _family_rows()
    receipt = qualification.qualify_decoded_rows(
        document_rows=documents,
        example_rows=examples,
        source_binding={"fixture": True},
        demands=_demands(2),
    )
    assert receipt["status"] == "qualified_source_capacity_no_selection"
    assert all(
        row["component_capacity"] == 2
        for row in receipt["family_aggregates"].values()
    )
    serialized = qualification.canonical_json(receipt).decode("ascii")
    assert "private query" not in serialized
    assert receipt["claim_boundary"]["reasoning_column_read"] is False
    assert receipt["claim_boundary"]["document_content_column_read"] is False
    _verify_hash(receipt)


def test_normalized_query_collision_reduces_component_capacity() -> None:
    documents, examples = _family_rows()
    examples["BIOLOGY"][1]["query"] = "  PRIVATE   QUERY biology 0  "
    receipt = qualification.qualify_decoded_rows(
        document_rows=documents,
        example_rows=examples,
        source_binding={},
        demands=_demands(2),
    )
    assert receipt["status"] == "terminal_source_infeasible_no_selection"
    biology = receipt["family_aggregates"]["BIOLOGY"]
    assert biology["component_capacity"] == 1
    assert biology["multirow_component_count"] == 1


def test_unknown_gold_is_aggregated_and_never_serialized() -> None:
    documents, examples = _family_rows()
    examples["ROBOTICS"][0]["gold_ids"] = ["secret-missing-document"]
    receipt = qualification.qualify_decoded_rows(
        document_rows=documents,
        example_rows=examples,
        source_binding={},
        demands={"BIOLOGY": 2, "ECONOMICS": 2, "ROBOTICS": 1},
    )
    robotics = receipt["family_aggregates"]["ROBOTICS"]
    assert robotics["eligible_example_count"] == 1
    assert robotics["example_invalid_reason_counts"] == {
        "gold_id_absent_from_documents": 1
    }
    assert "secret-missing-document" not in qualification.canonical_json(receipt).decode("ascii")


def test_duplicate_document_identifier_is_terminal() -> None:
    documents, examples = _family_rows()
    documents["ECONOMICS"].append({"id": "economics-doc-0"})
    receipt = qualification.qualify_decoded_rows(
        document_rows=documents,
        example_rows=examples,
        source_binding={},
        demands=_demands(2),
    )
    assert receipt["status"] == "terminal_source_infeasible_no_selection"
    assert receipt["family_aggregates"]["ECONOMICS"]["document_duplicate_id_count"] == 1


def test_present_excluded_gold_overlap_is_ineligible() -> None:
    documents, examples = _family_rows()
    examples["BIOLOGY"][0]["excluded_ids"] = ["biology-doc-0"]
    receipt = qualification.qualify_decoded_rows(
        document_rows=documents,
        example_rows=examples,
        source_binding={},
        demands={"BIOLOGY": 1, "ECONOMICS": 2, "ROBOTICS": 2},
    )
    assert receipt["family_aggregates"]["BIOLOGY"]["example_invalid_reason_counts"] == {
        "excluded_gold_overlap": 1
    }


def test_attempt_root_refuses_reuse(tmp_path) -> None:
    project = tmp_path
    root = project / qualification.ATTEMPT_ROOT_RELATIVE
    root.mkdir(parents=True)
    with pytest.raises(qualification.OneShotRefusal):
        qualification._create_attempt(
            project,
            {"self_sha256": "a" * 64},
        )
