from __future__ import annotations

import copy
import inspect
import json
import sqlite3
from typing import Any

import pytest

from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as adapter
from assumption_agent.benchmarks.feverous_p6_e2_acquisition_v1 import (
    CORPUS_UNIT_COUNT,
    CORPUS_VIEW_SCHEMA,
)
from assumption_agent.benchmarks.feverous_wikipedia_source_qualification_v1 import (
    FeverousWikiResolver,
    parse_element_id,
)


PAGE_A = "Synthetic_Official_Page_A"
PAGE_B = "Synthetic_Official_Page_B"


def _page_a() -> dict[str, Any]:
    return {
        "title": PAGE_A,
        "order": [
            "section_0",
            "sentence_0",
            "sentence_1",
            "table_0",
            "list_0",
        ],
        "section_0": {"value": "Section A", "level": 1},
        "sentence_0": "Alpha was founded in 1999.",
        "sentence_1": "Alpha is based in North.",
        "table_0": {
            "type": "normal",
            "caption": "Population table",
            "table": [
                [
                    {
                        "id": "header_cell_0_0_0",
                        "value": "Place",
                        "is_header": True,
                        "row_span": "1",
                        "column_span": "1",
                    },
                    {
                        "id": "header_cell_0_0_1",
                        "value": "2020",
                        "is_header": True,
                        "row_span": "1",
                        "column_span": "2",
                    },
                ],
                [
                    {
                        "id": "header_cell_0_1_0",
                        "value": "North",
                        "is_header": True,
                        "row_span": "1",
                        "column_span": "1",
                    },
                    {
                        "id": "cell_0_1_1",
                        "value": "12",
                        "is_header": False,
                        "row_span": "1",
                        "column_span": "1",
                    },
                    {
                        "id": "cell_0_1_2",
                        "value": " \u3000\t ",
                        "is_header": False,
                        "row_span": "1",
                        "column_span": "1",
                    },
                ],
            ],
        },
        "list_0": {
            "type": "unordered_list",
            "list": [
                {"id": "item_0_0", "value": "Parent item", "level": 0},
                {"id": "item_0_1", "value": "Child item", "level": 1},
            ],
        },
    }


def _page_b() -> dict[str, Any]:
    return {
        "title": PAGE_B,
        "order": ["section_0", "sentence_0", "sentence_1"],
        "section_0": {"value": "Section B", "level": 1},
        "sentence_0": "Beta acquired Alpha.",
        "sentence_1": "Beta is based in South.",
    }


def _connection(*pages: dict[str, Any]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE wiki (id TEXT PRIMARY KEY, data TEXT)")
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


def _full(page: str, local_id: str) -> str:
    return f"{page}_{local_id}"


def _evidence(*content_ids: str) -> dict[str, Any]:
    context: dict[str, list[str]] = {}
    for content_id in content_ids:
        parsed = parse_element_id(content_id)
        context[content_id] = [
            _full(parsed.page, "title"),
            _full(parsed.page, "section_0"),
        ]
    return {"content": list(content_ids), "context": context}


def _record(
    source_id: int,
    family: str,
    evidence: list[dict[str, Any]],
    *,
    claim: str | None = None,
    label: str = "SUPPORTS",
) -> dict[str, Any]:
    return {
        "annotator_operations": [],
        "challenge": family,
        "claim": claim or f"PRIVATE CLAIM {source_id}",
        "evidence": evidence,
        "id": source_id,
        "label": label,
    }


def _adapted_sources():
    page_a = _page_a()
    page_b = _page_b()
    connection = _connection(page_a, page_b)
    resolver = FeverousWikiResolver(connection)
    pages = {
        PAGE_A: adapter.adapt_qualified_page(
            PAGE_A,
            page_a,
            resolver=resolver,
            binding=adapter.FROZEN_TRAIN_BINDING,
        ),
        PAGE_B: adapter.adapt_qualified_page(
            PAGE_B,
            page_b,
            resolver=resolver,
            binding=adapter.FROZEN_TRAIN_BINDING,
        ),
    }
    return connection, resolver, pages


def test_exact_four_family_structural_eligibility_and_all_alternatives() -> None:
    connection, resolver, pages = _adapted_sources()
    sentence_a = _full(PAGE_A, "sentence_0")
    sentence_a2 = _full(PAGE_A, "sentence_1")
    sentence_b = _full(PAGE_B, "sentence_0")
    cell_a = _full(PAGE_A, "cell_0_1_1")
    header_a = _full(PAGE_A, "header_cell_0_0_1")
    caption_a = _full(PAGE_A, "table_caption_0")
    records = [
        _record(
            1,
            "Combining Tables and Text",
            [
                _evidence(sentence_a, cell_a),
                _evidence(sentence_a, caption_a),
            ],
        ),
        _record(
            2,
            "Entity Disambiguation",
            [_evidence(sentence_a, sentence_a2)],
            label="REFUTES",
        ),
        _record(
            3,
            "Multi-hop Reasoning",
            [_evidence(sentence_a, sentence_b)],
        ),
        _record(
            4,
            "Numerical Reasoning",
            [_evidence(cell_a, header_a)],
            label="REFUTES",
        ),
        {},
    ]
    batch = adapter.adapt_train_records(
        records,
        source_split="TRAIN",
        resolver=resolver,
        pages=pages,
        binding=adapter.FROZEN_TRAIN_BINDING,
    )
    assert [candidate.family for candidate in batch.candidates] == [
        "Combining Tables and Text",
        "Entity Disambiguation",
        "Multi-hop Reasoning",
        "Numerical Reasoning",
    ]
    combining = batch.candidates[0]
    assert set(combining.all_official_evidence_keys) == {
        sentence_a,
        cell_a,
        caption_a,
    }
    assert set(combining.evidence_sets) == {
        tuple(sorted((sentence_a, cell_a))),
        tuple(sorted((sentence_a, caption_a))),
    }
    assert batch.receipt["record_status_counts"] == {
        "blank_sentinel": 1,
        "eligible_candidate": 4,
    }
    assert batch.receipt["candidate_count"] == 4
    assert adapter.verify_adapter_receipt(batch.receipt)
    connection.close()


def test_empty_and_wrong_cardinality_sets_are_filtered_before_candidate() -> None:
    connection, resolver, pages = _adapted_sources()
    cell = _full(PAGE_A, "cell_0_1_1")
    empty = _full(PAGE_A, "cell_0_1_2")
    header = _full(PAGE_A, "header_cell_0_0_1")
    six = (
        _full(PAGE_A, "sentence_0"),
        _full(PAGE_A, "sentence_1"),
        _full(PAGE_A, "table_caption_0"),
        _full(PAGE_A, "header_cell_0_0_0"),
        _full(PAGE_A, "header_cell_0_0_1"),
        _full(PAGE_A, "item_0_0"),
    )
    record = _record(
        7,
        "Entity Disambiguation",
        [
            _evidence(empty, header),
            _evidence(*six),
            _evidence(cell, header),
        ],
    )
    decision = adapter.adapt_train_record(
        record,
        source_split="TRAIN",
        resolver=resolver,
        pages=pages,
        binding=adapter.FROZEN_TRAIN_BINDING,
    )
    assert decision.candidate is not None
    assert decision.candidate.evidence_sets == (tuple(sorted((cell, header))),)
    assert set(decision.candidate.all_official_evidence_keys) == {
        empty,
        cell,
        header,
        *six,
    }
    assert decision.excluded_empty_set_count == 1
    assert decision.excluded_cardinality_set_count == 1
    connection.close()


@pytest.mark.parametrize(
    ("family", "content"),
    [
        (
            "Combining Tables and Text",
            ("cell_0_1_1", "header_cell_0_0_1"),
        ),
        (
            "Multi-hop Reasoning",
            ("sentence_0", "sentence_1"),
        ),
        (
            "Numerical Reasoning",
            ("sentence_0", "sentence_1"),
        ),
    ],
)
def test_family_mismatch_never_becomes_a_canonical_candidate(
    family: str, content: tuple[str, str]
) -> None:
    connection, resolver, pages = _adapted_sources()
    record = _record(
        8,
        family,
        [_evidence(*(_full(PAGE_A, local_id) for local_id in content))],
    )
    decision = adapter.adapt_train_record(
        record,
        source_split="TRAIN",
        resolver=resolver,
        pages=pages,
        binding=adapter.FROZEN_TRAIN_BINDING,
    )
    assert decision.candidate is None
    assert decision.status == "no_eligible_canonical_set"
    assert decision.excluded_family_structure_set_count == 1
    connection.close()


def test_context_page_drift_fails_closed_without_guessing() -> None:
    connection, resolver, pages = _adapted_sources()
    content = _full(PAGE_A, "sentence_0")
    evidence = _evidence(content)
    evidence["context"][content] = [
        _full(PAGE_B, "title"),
        _full(PAGE_A, "section_0"),
    ]
    with pytest.raises(
        adapter.FeverousSourceAdapterError,
        match="page drifted",
    ):
        adapter.adapt_train_record(
            _record(9, "Entity Disambiguation", [evidence]),
            source_split="TRAIN",
            resolver=resolver,
            pages=pages,
            binding=adapter.FROZEN_TRAIN_BINDING,
        )
    connection.close()


def test_page_conversion_binds_resolver_atomic_compiler_and_corpus_schema() -> None:
    connection, _resolver, pages = _adapted_sources()
    adapted = pages[PAGE_A]
    assert _full(PAGE_A, "cell_0_1_2") in adapted.excluded_empty_unit_keys
    assert _full(PAGE_A, "cell_0_1_2") not in adapted.unit_by_key
    unit = adapted.unit_by_key[_full(PAGE_A, "cell_0_1_1")]
    assert unit.text.startswith("TARGET: 12\n")
    assert unit.page == PAGE_A
    assert unit.local_id == "cell_0_1_1"
    assert unit.sidecar["linearizer_version"] == adapter.ATOMIC_COMPILER_VERSION
    assert adapter.FROZEN_TRAIN_BINDING.corpus_view_schema == CORPUS_VIEW_SCHEMA
    assert adapter.FROZEN_TRAIN_BINDING.frozen_corpus_unit_count == CORPUS_UNIT_COUNT == 8192
    connection.close()


def test_dev_test_and_selection_interfaces_are_absent() -> None:
    connection, resolver, pages = _adapted_sources()
    with pytest.raises(
        adapter.FeverousSourceAdapterError,
        match="only the source-qualified official TRAIN",
    ):
        adapter.adapt_train_records(
            [],
            source_split="DEV",
            resolver=resolver,
            pages=pages,
            binding=adapter.FROZEN_TRAIN_BINDING,
        )
    assert not hasattr(adapter, "select_private_blocks")
    for function in (
        adapter.adapt_qualified_page,
        adapter.adapt_train_record,
        adapter.adapt_train_records,
    ):
        parameters = inspect.signature(function).parameters
        assert "secret" not in parameters
        assert "utilities" not in parameters
        assert "scores" not in parameters
    connection.close()


def test_aggregate_receipt_contains_no_raw_claim_page_or_evidence() -> None:
    connection, resolver, pages = _adapted_sources()
    claim = "PRIVATE UNMISTAKABLE CLAIM CONTENT"
    evidence_id = _full(PAGE_A, "sentence_0")
    other_id = _full(PAGE_A, "sentence_1")
    batch = adapter.adapt_train_records(
        [
            _record(
                424242,
                "Entity Disambiguation",
                [_evidence(evidence_id, other_id)],
                claim=claim,
            )
        ],
        source_split="TRAIN",
        resolver=resolver,
        pages=pages,
        binding=adapter.FROZEN_TRAIN_BINDING,
    )
    serialized = json.dumps(dict(batch.receipt), sort_keys=True)
    for forbidden in (
        claim,
        evidence_id,
        other_id,
        PAGE_A,
        "424242",
        "Alpha was founded",
    ):
        assert forbidden not in serialized
    assert batch.receipt["raw_claim_page_or_evidence_serialized"] is False
    assert batch.receipt["cohort_block_or_canonical_set_selected"] is False
    assert batch.receipt["fixed_8192_corpus_formed"] is False
    assert batch.receipt["utility_recipe_or_model_accessed"] is False

    tampered = copy.deepcopy(dict(batch.receipt))
    tampered["candidate_count"] = 999
    with pytest.raises(adapter.FeverousSourceAdapterError, match="drifted"):
        adapter.verify_adapter_receipt(tampered)
    connection.close()


def test_candidate_screen_and_corpus_compile_have_bounded_separate_streams() -> None:
    page_a = _page_a()
    page_b = _page_b()
    connection = _connection(page_a, page_b)
    resolver = FeverousWikiResolver(connection)
    sentence_a = _full(PAGE_A, "sentence_0")
    sentence_b = _full(PAGE_B, "sentence_0")
    candidate_batch = adapter.adapt_train_candidate_records(
        [
            _record(
                55,
                "Multi-hop Reasoning",
                [_evidence(sentence_a, sentence_b)],
            )
        ],
        source_split="TRAIN",
        resolver=resolver,
        binding=adapter.FROZEN_TRAIN_BINDING,
    )
    assert len(candidate_batch.candidates) == 1
    assert candidate_batch.corpus_units == ()
    # Page-local resolver windows are discarded rather than retained by the
    # caller's resolver across the complete candidate scan.
    assert resolver.lookup_count == 0

    yielded_pages: list[str] = []

    def page_rows():
        yielded_pages.append(PAGE_A)
        yield PAGE_A, page_a
        yielded_pages.append(PAGE_B)
        yield PAGE_B, page_b

    stream = adapter.iter_qualified_corpus_units(
        page_rows(),
        resolver=resolver,
        binding=adapter.FROZEN_TRAIN_BINDING,
    )
    assert yielded_pages == []
    with pytest.raises(
        adapter.FeverousSourceAdapterError,
        match="before normal exhaustion",
    ):
        stream.aggregate_receipt()
    first = next(stream)
    assert first.page == PAGE_A
    assert yielded_pages == [PAGE_A]
    units = [first, *stream]
    assert {unit.page for unit in units} == {PAGE_A, PAGE_B}
    receipt = stream.aggregate_receipt()
    assert receipt["adapted_page_count"] == 2
    assert receipt["emitted_eligible_atomic_unit_count"] == len(units)
    assert receipt["excluded_empty_atomic_unit_count"] == 1
    assert receipt["maximum_resident_compiled_pages"] == 1
    assert receipt["all_units_materialized_or_sorted"] is False
    assert adapter.verify_corpus_stream_receipt(receipt)
    # The base resolver still has no accumulated 38k-page cache.
    assert resolver.lookup_count == 0
    serialized = json.dumps(dict(receipt), sort_keys=True)
    assert PAGE_A not in serialized and PAGE_B not in serialized
    connection.close()
