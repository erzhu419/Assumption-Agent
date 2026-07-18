from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    feverous_wikipedia_source_qualification_v1 as qualification,
)


PAGE = "PRIVATE_Page_with_header_cell_and_many_underscores"
OTHER_PAGE = "PRIVATE_Other_page_with_underscores"


def _page(page: str = PAGE) -> dict[str, Any]:
    return {
        "title": page,
        "order": [
            "section_0",
            "sentence_0",
            "table_0",
            "list_0",
        ],
        "section_0": {"value": "PRIVATE_SECTION_TEXT", "level": 1},
        "sentence_0": "PRIVATE_SENTENCE_TEXT",
        "table_0": {
            "type": "normal",
            "caption": "PRIVATE_TABLE_CAPTION",
            "table": [
                [
                    {
                        # Coordinates are intentionally unrelated to the
                        # physical list position.  Resolution must use id.
                        "id": "header_cell_0_9_4",
                        "value": "PRIVATE_HEADER_TEXT",
                        "is_header": True,
                        "row_span": 2,
                        "column_span": 3,
                    },
                    {
                        "id": "cell_0_42_7",
                        "value": "PRIVATE_CELL_TEXT",
                        "is_header": False,
                        "row_span": 2,
                        "column_span": 1,
                    },
                    {
                        "id": "header_cell_0_8_8",
                        "value": "PRIVATE_WRONG_HEADER_TEXT",
                        "is_header": False,
                        "row_span": 1,
                        "column_span": 1,
                    },
                ],
                [
                    {
                        "id": "cell_0_6_6",
                        "value": "PRIVATE_AMBIGUOUS_A",
                        "is_header": False,
                        "row_span": 1,
                        "column_span": 1,
                    },
                    {
                        "id": "cell_0_6_6",
                        "value": "PRIVATE_AMBIGUOUS_B",
                        "is_header": False,
                        "row_span": 1,
                        "column_span": 1,
                    },
                ],
            ],
        },
        "list_0": {
            "type": "unordered_list",
            "list": [
                {
                    "id": "item_0_7",
                    "value": "PRIVATE_LIST_ITEM_TEXT",
                    "level": 0,
                }
            ],
        },
    }


def _connection(*pages: dict[str, Any]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    # Deliberately no uniqueness constraint: exact duplicate wiki rows must
    # be observed as ambiguous rather than silently taking the first row.
    connection.execute("CREATE TABLE wiki (id TEXT, data TEXT)")
    connection.executemany(
        "INSERT INTO wiki (id, data) VALUES (?, ?)",
        [
            (
                page["title"],
                json.dumps(page, ensure_ascii=False, separators=(",", ":")),
            )
            for page in pages
        ],
    )
    connection.commit()
    connection.execute("PRAGMA query_only = ON")
    return connection


def _valid_evidence() -> list[dict[str, Any]]:
    def full(local_id: str) -> str:
        return f"{PAGE}_{local_id}"

    content = [
        full("sentence_0"),
        full("cell_0_42_7"),
        full("header_cell_0_9_4"),
        full("item_0_7"),
        full("table_caption_0"),
    ]
    return [
        {
            "content": content,
            "context": {
                content[0]: [full("title"), full("section_0")],
                content[1]: [full("title"), full("header_cell_0_9_4")],
                content[2]: [full("title")],
                content[3]: [full("title"), full("section_0")],
                content[4]: [full("title")],
            },
        }
    ]


def test_longest_suffix_and_fixed_arity_preserve_underscored_page() -> None:
    parsed = qualification.parse_element_id(
        f"{PAGE}_header_cell_0_9_4"
    )
    assert parsed.page == PAGE
    assert parsed.kind == "header_cell"
    assert parsed.indices == (0, 9, 4)
    assert parsed.local_id == "header_cell_0_9_4"

    page_with_kind_text = qualification.parse_element_id(
        "page_sentence_3_header_cell_name_cell_0_12_8"
    )
    assert page_with_kind_text.page == "page_sentence_3_header_cell_name"
    assert page_with_kind_text.kind == "cell"
    with pytest.raises(
        qualification.FeverousWikipediaQualificationError,
        match="fixed-arity",
    ):
        qualification.parse_element_id(f"{PAGE}_header_cell_0_9")
    with pytest.raises(qualification.FeverousWikipediaQualificationError):
        qualification.parse_element_id(f"{PAGE}_sentence_0_1")


def test_valid_topology_resolves_exact_ids_and_spans_content_free() -> None:
    connection = _connection(_page())
    resolver = qualification.FeverousWikiResolver(connection)
    cell = resolver.resolve_exact(
        f"{PAGE}_cell_0_42_7",
        context_page=PAGE,
    )
    assert cell.status == "resolved"
    assert cell.element is not None
    assert cell.element.value == "PRIVATE_CELL_TEXT"
    assert cell.element.indices == (0, 42, 7)
    assert cell.element.row_span == 2
    assert cell.element.column_span == 1

    receipt = qualification.qualify_evidence_sets(
        _valid_evidence(),
        connection,
        database_size_bytes=123,
        database_sha256="1" * 64,
        archive_size_bytes=456,
        archive_sha256="2" * 64,
    )
    assert receipt["status"] == (
        "passed_exact_source_qualification_no_selection"
    )
    assert receipt["evidence_aggregate"]["content_kind_counts"] == {
        "cell": 1,
        "header_cell": 1,
        "item": 1,
        "sentence": 1,
        "table_caption": 1,
    }
    assert receipt["context_exactness"] == {
        "exact_context_key_count": 5,
        "missing_content_context_key_count": 0,
        "orphan_context_key_count": 0,
        "missing_title_context_count": 0,
        "ambiguous_title_context_count": 0,
        "content_title_page_drift_count": 0,
        "context_member_title_page_drift_count": 0,
        "fuzzy_lookup_or_repair_count": 0,
    }
    resolution = receipt["element_resolution"]
    assert resolution["row_span_gt_one_cell_count"] == 2
    assert resolution["column_span_gt_one_cell_count"] == 1
    assert resolution[
        "cell_resolution_uses_exact_cell_id_not_coordinates"
    ] is True
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for private_value in (
        "PRIVATE_",
        "header_cell_and_many_underscores",
        "SENTENCE_TEXT",
        "CELL_TEXT",
        "TABLE_CAPTION",
        "LIST_ITEM_TEXT",
    ):
        assert private_value not in serialized
    assert receipt["claim_boundary"][
        "identifiers_titles_or_text_serialized"
    ] is False
    assert qualification.verify_receipt(receipt) is True
    changed = deepcopy(receipt)
    changed["evidence_aggregate"]["content_reference_count"] = 999
    assert qualification.verify_receipt(changed) is False
    connection.close()


def test_official_decimal_string_spans_are_losslessly_accepted() -> None:
    page = _page()
    table = page["table_0"]
    assert isinstance(table, dict)
    rows = table["table"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, list)
        for cell in row:
            assert isinstance(cell, dict)
            cell["row_span"] = str(cell["row_span"])
            cell["column_span"] = str(cell["column_span"])
    connection = _connection(page)
    resolver = qualification.FeverousWikiResolver(connection)
    resolution = resolver.resolve_exact(
        f"{PAGE}_cell_0_42_7", context_page=PAGE
    )
    assert resolution.status == "resolved"
    assert resolution.element is not None
    assert resolution.element.row_span == 2
    assert resolution.element.column_span == 1
    connection.close()


def test_missing_ambiguous_and_wrong_header_are_distinct() -> None:
    connection = _connection(_page())
    title = f"{PAGE}_title"
    missing = f"{PAGE}_sentence_99"
    ambiguous = f"{PAGE}_cell_0_6_6"
    wrong_header = f"{PAGE}_header_cell_0_8_8"
    evidence = [
        {
            "content": [missing, ambiguous, wrong_header],
            "context": {
                missing: [title],
                ambiguous: [title],
                wrong_header: [title],
            },
        }
    ]
    receipt = qualification.qualify_evidence_sets(evidence, connection)
    assert receipt["status"] == (
        "failed_exact_source_qualification_no_selection"
    )
    assert receipt["element_resolution"]["content_status_counts"] == {
        "ambiguous": 1,
        "missing": 1,
        "wrong_header": 1,
    }
    assert receipt["context_exactness"]["fuzzy_lookup_or_repair_count"] == 0
    connection.close()


def test_context_title_is_the_only_page_authority_and_drifts_are_separate() -> None:
    connection = _connection(_page(), _page(OTHER_PAGE))
    content = f"{OTHER_PAGE}_sentence_0"
    evidence = [
        {
            "content": [content],
            "context": {
                content: [
                    f"{PAGE}_title",
                    f"{OTHER_PAGE}_section_0",
                ]
            },
        }
    ]
    receipt = qualification.qualify_evidence_sets(evidence, connection)
    exactness = receipt["context_exactness"]
    assert exactness["content_title_page_drift_count"] == 1
    assert exactness["context_member_title_page_drift_count"] == 1
    assert exactness["fuzzy_lookup_or_repair_count"] == 0
    assert receipt["element_resolution"]["content_status_counts"] == {
        "wrong_page": 1
    }
    assert receipt["element_resolution"]["context_status_counts"] == {
        "resolved": 1,
        "wrong_page": 1,
    }
    # Only PAGE was looked up.  The content id must not redirect the resolver
    # to OTHER_PAGE after the exact context title selected PAGE.
    assert receipt["page_resolution"]["exact_lookup_count"] == 1
    connection.close()


def test_missing_or_ambiguous_title_context_does_not_guess() -> None:
    connection = _connection(_page(), _page(OTHER_PAGE))
    content_a = f"{PAGE}_sentence_0"
    content_b = f"{PAGE}_cell_0_42_7"
    evidence = [
        {
            "content": [content_a, content_b],
            "context": {
                content_a: [f"{PAGE}_section_0"],
                content_b: [f"{PAGE}_title", f"{OTHER_PAGE}_title"],
            },
        }
    ]
    receipt = qualification.qualify_evidence_sets(evidence, connection)
    exactness = receipt["context_exactness"]
    assert exactness["missing_title_context_count"] == 1
    assert exactness["ambiguous_title_context_count"] == 1
    assert receipt["page_resolution"]["exact_lookup_count"] == 0
    connection.close()


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(
        qualification.FeverousWikipediaQualificationError,
        match="duplicate JSON object key",
    ):
        qualification._decode_strict_json('{"title":"a","title":"b"}')
    with pytest.raises(
        qualification.FeverousWikipediaQualificationError,
        match="non-finite",
    ):
        qualification._decode_strict_json('{"value":NaN}')


def test_page_row_is_exact_and_ambiguous_rows_are_not_first_wins() -> None:
    page = _page()
    connection = _connection(page, deepcopy(page))
    content = f"{PAGE}_sentence_0"
    evidence = [
        {
            "content": [content],
            "context": {content: [f"{PAGE}_title"]},
        }
    ]
    receipt = qualification.qualify_evidence_sets(evidence, connection)
    assert receipt["page_resolution"]["status_counts"] == {"ambiguous": 1}
    assert receipt["element_resolution"]["content_status_counts"] == {
        "page_ambiguous": 1
    }
    connection.close()


def test_formal_opener_is_immutable_and_query_only(tmp_path) -> None:
    database = tmp_path / "synthetic.db"
    writable = sqlite3.connect(database)
    writable.execute("CREATE TABLE wiki (id TEXT PRIMARY KEY, data TEXT)")
    page = _page()
    writable.execute(
        "INSERT INTO wiki (id, data) VALUES (?, ?)",
        (PAGE, json.dumps(page, ensure_ascii=False)),
    )
    writable.commit()
    writable.close()

    connection = qualification.open_immutable_wiki_db(database)
    assert connection.execute("PRAGMA query_only").fetchone() == (1,)
    assert connection.execute("SELECT COUNT(*) FROM wiki").fetchone() == (1,)
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM wiki")
    connection.close()


def test_qualification_refuses_non_query_only_connection() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE wiki (id TEXT, data TEXT)")
    with pytest.raises(
        qualification.FeverousWikipediaQualificationError,
        match="not query_only",
    ):
        qualification.qualify_evidence_sets([], connection)
    connection.close()


def test_public_source_qualification_manifest_self_hashes() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "feverous_wikipedia_source_qualification_v1.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert qualification.verify_receipt(receipt)
    assert receipt["status"] == (
        "passed_exact_wikipedia_source_qualification_no_selection"
    )
