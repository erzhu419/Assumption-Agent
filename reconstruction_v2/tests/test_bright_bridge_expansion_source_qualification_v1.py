from __future__ import annotations

from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_bridge_expansion_source_qualification_v1 as qualifier,
)


def _rows(count: int) -> tuple[dict[str, object], dict[str, object]]:
    documents = tuple({"id": f"doc-{index}"} for index in range(count + 5))
    examples = tuple(
        {
            "query": f"query {index}",
            "id": f"item-{index}",
            "excluded_ids": [],
            "gold_ids": [f"doc-{index}"],
        }
        for index in range(count)
    )
    return documents, examples


def _qualified_inputs(count: int = 68):
    document_rows = {}
    example_rows = {}
    expected_counts = {}
    for family in qualifier.FAMILY_ORDER:
        documents, examples = _rows(count)
        document_rows[family] = documents
        example_rows[family] = examples
        expected_counts[family] = {
            "documents": len(documents),
            "examples": len(examples),
        }
    return document_rows, example_rows, expected_counts


def test_activate_replaces_every_epoch_binding() -> None:
    qualifier._activate()
    assert qualifier.base.SCHEMA == qualifier.SCHEMA
    assert qualifier.base.FAMILY_ORDER == (
        "PONY",
        "PSYCHOLOGY",
        "SUSTAINABLE_LIVING",
    )
    assert qualifier.base.DEMANDS == {
        "PONY": 68,
        "PSYCHOLOGY": 68,
        "SUSTAINABLE_LIVING": 68,
    }
    assert qualifier.base.SOURCE_ROOT_RELATIVE == Path(
        "artifacts/bright_bridge_expansion_source_v1/dataset"
    )


def test_exact_68_per_family_qualifies_without_selection() -> None:
    document_rows, example_rows, expected_counts = _qualified_inputs()
    receipt = qualifier.qualify_decoded_rows(
        document_rows=document_rows,
        example_rows=example_rows,
        source_binding={"synthetic": True},
        expected_counts=expected_counts,
    )
    assert receipt["schema"] == qualifier.SCHEMA
    assert receipt["status"] == "qualified_source_capacity_no_selection"
    assert receipt["claim_boundary"]["reasoning_column_read"] is False
    assert receipt["claim_boundary"]["document_content_column_read"] is False


def test_67_in_one_family_is_terminal_source_infeasible() -> None:
    document_rows, example_rows, expected_counts = _qualified_inputs()
    documents, examples = _rows(67)
    example_rows["PONY"] = examples
    document_rows["PONY"] = documents
    expected_counts["PONY"] = {
        "documents": len(documents),
        "examples": len(examples),
    }
    receipt = qualifier.qualify_decoded_rows(
        document_rows=document_rows,
        example_rows=example_rows,
        source_binding={"synthetic": True},
        expected_counts=expected_counts,
    )
    assert receipt["status"] == "terminal_source_infeasible_no_selection"
    assert receipt["family_aggregates"]["PONY"]["component_capacity"] == 67


def test_gold_outside_corpus_is_excluded_not_printed() -> None:
    document_rows, example_rows, expected_counts = _qualified_inputs()
    example_rows["PONY"][0]["gold_ids"] = ["missing-document"]
    receipt = qualifier.qualify_decoded_rows(
        document_rows=document_rows,
        example_rows=example_rows,
        source_binding={"synthetic": True},
        expected_counts=expected_counts,
    )
    assert (
        receipt["family_aggregates"]["PONY"]["example_invalid_reason_counts"]
        == {"gold_id_absent_from_documents": 1}
    )
    assert "missing-document" not in qualifier.canonical_json(receipt).decode("ascii")


def test_source_bindings_match_pinned_metadata() -> None:
    assert qualifier.SOURCE_BINDINGS["PONY"]["documents"]["rows"] == 7_894
    assert qualifier.SOURCE_BINDINGS["PSYCHOLOGY"]["examples"]["rows"] == 101
    assert (
        qualifier.SOURCE_BINDINGS["SUSTAINABLE_LIVING"]["documents"]["sha256"]
        == "474628623cf9de252bd80a7d1b667aa5070e21b87e1dd33f6723db4d24121fdf"
    )


def test_base_implementation_binding_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / qualifier.BASE_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(qualifier.BrightQualificationError):
        qualifier._verify_base_binding(tmp_path)
