from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from assumption_agent.benchmarks import feverous_atomic_corpus_v1 as atomic


PAGE = "Synthetic_FEVEROUS_Page"


def _page() -> dict[str, Any]:
    return {
        "title": PAGE,
        "order": [
            "section_0",
            "sentence_0",
            "section_1",
            "table_0",
            "list_0",
        ],
        "section_0": {"value": "  Main\u3000section ", "level": 1},
        "sentence_0": "  A\u3000target   sentence.  ",
        "section_1": {"value": "Details", "level": 2},
        "table_0": {
            "type": "normal",
            "caption": "  Population\u3000by year ",
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
                        "row_span": "2",
                        "column_span": "1",
                    },
                    {
                        "id": "cell_0_1_1",
                        "value": "10",
                        "is_header": False,
                        "row_span": "1",
                        "column_span": "1",
                    },
                    {
                        "id": "cell_0_1_2",
                        "value": "11",
                        "is_header": False,
                        "row_span": "1",
                        "column_span": "1",
                    },
                ],
                [
                    {
                        "id": "cell_0_2_1",
                        "value": "  12\u00a0people ",
                        "is_header": False,
                        "row_span": "1",
                        "column_span": "1",
                    },
                    {
                        "id": "cell_0_2_2",
                        "value": "13",
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
                {"id": "item_0_0", "value": "Parent", "level": 0},
                {"id": "item_0_1", "value": " Child ", "level": 1},
            ],
        },
    }


def _unit(compilation: atomic.PageCompilation, local_id: str) -> atomic.AtomicUnit:
    return next(unit for unit in compilation.units if unit.sidecar.local_id == local_id)


def test_lightweight_identity_enumerator_matches_full_compile_and_commits_no_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enumeration = atomic.enumerate_official_page_atomic_identities(PAGE, _page())
    compiled = atomic.compile_official_page(PAGE, _page())
    assert atomic.crosscheck_identity_enumeration(enumeration, compiled) is compiled
    assert [row.local_id for row in enumeration.identities] == [
        unit.sidecar.local_id for unit in compiled.units
    ]
    assert [row.normalized_target for row in enumeration.identities] == [
        unit.target for unit in compiled.units
    ]
    assert all(len(row.target_sha256) == 64 for row in enumeration.identities)
    commitment = enumeration.commitment()
    assert set(commitment) == {
        "enumerator_version",
        "excluded_empty_count",
        "identity_count",
        "identity_enumeration_sha256",
        "schema",
    }
    assert commitment["identity_count"] == len(compiled.units)
    assert "normalized_target" not in commitment
    assert "page" not in commitment

    # The first pass must not accidentally regress to full context rendering.
    monkeypatch.setattr(
        atomic,
        "_render_parts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity pass rendered full text")
        ),
    )
    monkeypatch.setattr(
        atomic,
        "_header_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity pass built cell context")
        ),
    )
    repeated = atomic.enumerate_official_page_atomic_identities(PAGE, _page())
    assert repeated.commitment() == commitment


def test_identity_crosscheck_rejects_target_drift() -> None:
    enumeration = atomic.enumerate_official_page_atomic_identities(PAGE, _page())
    first = enumeration.identities[0]
    tampered_first = atomic.AtomicIdentity(
        page=first.page,
        local_id=first.local_id,
        unit_type=first.unit_type,
        official_ordinal=first.official_ordinal,
        normalized_target=first.normalized_target + " drift",
        target_sha256=first.target_sha256,
    )
    tampered = atomic.PageIdentityEnumeration(
        page=enumeration.page,
        identities=(tampered_first, *enumeration.identities[1:]),
        excluded_empty_local_ids=enumeration.excluded_empty_local_ids,
    )
    with pytest.raises(atomic.FeverousAtomicCorpusError, match="differ"):
        atomic.crosscheck_identity_enumeration(
            tampered,
            atomic.compile_official_page(PAGE, _page()),
        )


def test_official_page_compiles_target_first_exact_typed_sidecars() -> None:
    compiled = atomic.compile_official_page(PAGE, _page())
    sentence = _unit(compiled, "sentence_0")
    assert sentence.target == "A target sentence."
    assert sentence.text.startswith(
        "TARGET: A target sentence.\n"
        "TITLE: Synthetic_FEVEROUS_Page\n"
        "SECTION_PATH: Main section\n"
        "TYPE: sentence"
    )
    assert sentence.sidecar.page == PAGE
    assert sentence.sidecar.unit_type == "sentence"
    assert sentence.sidecar.coordinates == (0,)
    assert sentence.sidecar.section_ids == ("section_0",)
    assert sentence.sidecar.section_path == ("Main section",)

    caption = _unit(compiled, "table_caption_0")
    assert caption.text.startswith("TARGET: Population by year\n")
    assert caption.text.endswith("TABLE_KIND: normal")
    assert caption.sidecar.coordinates == (0,)
    assert caption.sidecar.section_ids == ("section_0", "section_1")

    item = _unit(compiled, "item_0_1")
    assert item.text.startswith("TARGET: Child\n")
    assert item.text.endswith("LIST_ANCESTOR_PATH: Parent")
    assert item.sidecar.list_ancestor_ids == ("item_0_0",)


def test_cell_header_axes_rowspan_and_marked_row_are_exact() -> None:
    compiled = atomic.compile_official_page(PAGE, _page())
    target = _unit(compiled, "cell_0_2_1")
    assert target.sidecar.coordinates == (0, 2, 1)
    assert target.sidecar.row_span == 1
    assert target.sidecar.column_span == 1
    # The row header starts one row earlier but its rowspan covers target row.
    assert target.sidecar.applicable_row_header_ids == (
        "header_cell_0_1_0",
    )
    # The column header's colspan covers the exact target column.
    assert target.sidecar.applicable_column_header_ids == (
        "header_cell_0_0_1",
    )
    assert "TABLE_CAPTION: Population by year" in target.text
    assert "APPLICABLE_HEADERS: ROW[North] COLUMN[2020]" in target.text
    assert (
        "ROW_WITH_TARGET_MARKED: North | <<TARGET>> 12 people <</TARGET>> | 13"
        in target.text
    )

    row_header = _unit(compiled, "header_cell_0_1_0")
    assert row_header.sidecar.unit_type == "header_cell"
    # FEVEROUS 0.54 binds a span-expanded id to its final row-major grid cell.
    assert row_header.sidecar.coordinates == (0, 2, 0)
    assert row_header.sidecar.row_span == 2


def test_table_topology_uses_official_span_grid_not_id_suffix_coordinates() -> None:
    page = {
        "title": PAGE,
        "order": ["table_0"],
        "table_0": {
            "type": "normal",
            "caption": "Counterfactual coordinates",
            "table": [
                [
                    {
                        "id": "header_cell_0_91_81",
                        "value": "Place",
                        "is_header": True,
                        "row_span": "1",
                        "column_span": "1",
                    },
                    {
                        "id": "header_cell_0_71_61",
                        "value": "Year",
                        "is_header": True,
                        "row_span": "1",
                        "column_span": "1",
                    },
                ],
                [
                    {
                        "id": "header_cell_0_51_41",
                        "value": "North",
                        "is_header": True,
                        "row_span": "1",
                        "column_span": "1",
                    },
                    {
                        "id": "cell_0_31_21",
                        "value": "12",
                        "is_header": False,
                        "row_span": "1",
                        "column_span": "1",
                    },
                ],
            ],
        },
    }
    compiled = atomic.compile_official_page(PAGE, page)
    target = _unit(compiled, "cell_0_31_21")
    assert target.sidecar.coordinates == (0, 1, 1)
    assert target.sidecar.applicable_row_header_ids == (
        "header_cell_0_51_41",
    )
    assert target.sidecar.applicable_column_header_ids == (
        "header_cell_0_71_61",
    )
    assert "ROW[North] COLUMN[Year]" in target.text


def test_all_three_arms_receive_the_same_bytes_and_agent_sidecar_is_separate() -> None:
    target = _unit(atomic.compile_official_page(PAGE, _page()), "cell_0_2_1")
    observed = tuple(target.bytes_for_arm(arm) for arm in atomic.ARM_IDS)
    assert observed[0] == observed[1] == observed[2] == target.text.encode("utf-8")
    assert b"applicable_row_header_ids" not in target.text_utf8
    with pytest.raises(atomic.FeverousAtomicCorpusError, match="unknown corpus arm"):
        target.bytes_for_arm("unregistered")


def test_target_first_serialization_is_tail_truncation_ready() -> None:
    target = _unit(atomic.compile_official_page(PAGE, _page()), "cell_0_2_1")
    first_line = target.text.splitlines()[0]
    assert first_line == "TARGET: 12 people"
    assert target.text.index("TARGET:") == 0
    token_ids = tuple(range(400))
    assert atomic.tail_truncate_token_ids(token_ids) == tuple(range(256))
    assert atomic.tail_truncate_token_ids(token_ids, maximum=3) == (0, 1, 2)


def test_empty_atomic_targets_are_explicitly_rejected_or_excluded() -> None:
    page = deepcopy(_page())
    page["sentence_0"] = " \u3000\t "
    table = page["table_0"]
    assert isinstance(table, dict)
    rows = table["table"]
    assert isinstance(rows, list)
    rows[2][1]["value"] = "\n\t"
    compiled = atomic.compile_official_page(PAGE, page)
    assert compiled.excluded_empty_local_ids == ("sentence_0", "cell_0_2_2")
    assert all(unit.target for unit in compiled.units)
    # Empty atoms retain an ordinal gap; neither empty deletion nor the section
    # boundary silently creates a false official-adjacency witness.
    table_units = [
        unit for unit in compiled.units if unit.sidecar.table_id == "table_0"
    ]
    assert min(unit.sidecar.official_ordinal for unit in table_units) > 0
    with pytest.raises(atomic.FeverousAtomicCorpusError, match="empty"):
        atomic.require_nonempty_atomic_target(" \u3000\n ")


def _span(claim: str, text: str, start: int = 0) -> atomic.NerSpan:
    left = claim.index(text, start)
    return atomic.NerSpan(left, left + len(text))


def test_claim_facet_compiler_uses_source_order_and_fixed_4_2_2_limits() -> None:
    claim = (
        "Alice founded Acme in 1999, Bob moved it on January 2, 2001; "
        "Carol paid $3.5 million to Delta in 2024. Echo agreed."
    )
    spans = tuple(_span(claim, name) for name in ("Alice", "Acme", "Bob", "Carol", "Delta", "Echo"))
    compiled = atomic.compile_claim_facets(claim, spans)
    assert [facet.text for facet in compiled.of_kind("entity")] == [
        "Alice",
        "Acme",
        "Bob",
        "Carol",
    ]
    assert [facet.text for facet in compiled.of_kind("numeric_or_date")] == [
        "1999",
        "January 2, 2001",
    ]
    relations = compiled.of_kind("relation_clause")
    assert len(relations) == 2
    assert relations[0].text == "[ENTITY] founded [ENTITY] in [NUMBER]"
    assert relations[1].text == "[ENTITY] moved it on [NUMBER]"
    assert compiled.normalized_claim[
        relations[1].source_start : relations[1].source_end
    ] == "Bob moved it on January 2, 2001"
    assert all(
        left.source_start <= right.source_start
        for left, right in zip(relations, relations[1:])
    )


def test_claim_nfkc_offsets_counterfactual_and_forbidden_record_fields() -> None:
    claim = "Ａlice\u3000founded   Acme in 1999."
    normalized = atomic.normalize_surface(claim)
    assert normalized == "Alice founded Acme in 1999."
    spans = (_span(normalized, "Alice"), _span(normalized, "Acme"))
    first = atomic.compile_claim_facets(claim, spans)

    # Metadata counterfactuals cannot enter because only the claim string is
    # passed across the compiler boundary.
    record_a = {
        "claim": claim,
        "label": "SUPPORTS",
        "challenge": "Numerical Reasoning",
        "evidence": {"forbidden": "A"},
    }
    record_b = {
        "claim": claim,
        "label": "REFUTES",
        "challenge": "Entity Disambiguation",
        "evidence": {"forbidden": "B"},
    }
    assert atomic.compile_claim_facets(record_a["claim"], spans) == first
    assert atomic.compile_claim_facets(record_b["claim"], spans) == first

    changed = "Diana founded Acme in 1999."
    changed_spans = (_span(changed, "Diana"), _span(changed, "Acme"))
    assert atomic.compile_claim_facets(changed, changed_spans) != first

    with pytest.raises(atomic.FeverousAtomicCorpusError, match="not a record"):
        atomic.compile_claim_facets(record_a)  # type: ignore[arg-type]
    with pytest.raises(atomic.FeverousAtomicCorpusError, match="offsets only"):
        atomic.compile_claim_facets(
            normalized,
            [{"start": 0, "end": 5, "label": "PER"}],  # type: ignore[list-item]
        )
