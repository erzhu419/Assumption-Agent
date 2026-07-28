from __future__ import annotations

import hashlib
import inspect

import pytest

from assumption_agent.benchmarks import quac_p1_action_adapter_v1 as adapter
from assumption_agent.benchmarks import quac_rjmc_evaluator_v1 as evaluator
from replication_runtime.quac_p1_official_v1 import contract as official_contract


def _basis(index: int, scale: float = 1.0) -> tuple[float, ...]:
    row = [0.0] * adapter.MINILM_EMBEDDING_DIMENSION
    row[index] = scale
    return tuple(row)


def _opaque(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _embedding_rows(
    documents: tuple[adapter.BlockDocument, ...],
    turns: tuple[adapter.QuestionTurn, ...],
    vectors_by_text: dict[str, tuple[float, ...]],
) -> tuple[adapter.MiniLmEmbedding, ...]:
    rows = []
    for digest, text in adapter.required_embedding_serializations(
        documents,
        turns,
    ):
        rows.append(
            adapter.MiniLmEmbedding(
                serialization_sha256=digest,
                vector=vectors_by_text[text],
            )
        )
    return tuple(rows)


def _maximal_fixture(
) -> tuple[
    tuple[adapter.BlockDocument, ...],
    tuple[adapter.QuestionTurn, ...],
    tuple[adapter.MiniLmEmbedding, ...],
]:
    documents: list[adapter.BlockDocument] = []
    for index in range(5):
        documents.append(
            adapter.BlockDocument(
                unit_id=_opaque(f"unit:r{index}"),
                context_id=_opaque(f"context:raw:{index}"),
                title=f"raw title {index}",
                section_title=f"raw section {index}",
                context_window_ordinal=0,
                text=f"lowercase raw evidence {index}",
            )
        )
    for index in range(4):
        documents.append(
            adapter.BlockDocument(
                unit_id=_opaque(f"unit:a{index}"),
                context_id=(
                    _opaque("context:expansion")
                    if index == 0
                    else _opaque(f"context:anchor:{index}")
                ),
                title=f"anchor title {index}",
                section_title=f"anchor section {index}",
                context_window_ordinal=0,
                text=f"lowercase anchor evidence {index}",
            )
        )
    documents.extend(
        (
            adapter.BlockDocument(
                unit_id=_opaque("unit:x0"),
                context_id=_opaque("context:expansion"),
                title="outside title zero",
                section_title="outside section zero",
                context_window_ordinal=1,
                text="lowercase first expansion",
            ),
            adapter.BlockDocument(
                unit_id=_opaque("unit:x1"),
                context_id=_opaque("context:expansion"),
                title="outside title one",
                section_title="outside section one",
                context_window_ordinal=2,
                text="lowercase second expansion",
            ),
        )
    )
    document_rows = tuple(documents)
    turns = tuple(
        adapter.QuestionTurn(f"question {slot}")
        for slot in range(adapter.MAX_DIALOGUE_TURNS)
    )

    vector_by_unit = {
        **{
            _opaque(f"unit:r{index}"): _basis(0)
            for index in range(5)
        },
        **{
            _opaque(f"unit:a{index}"): _basis(index + 1)
            for index in range(4)
        },
        _opaque("unit:x0"): _basis(5),
        _opaque("unit:x1"): _basis(6),
    }
    vectors_by_text = {
        adapter.serialize_evidence_unit(row): vector_by_unit[row.unit_id]
        for row in document_rows
    }
    vectors_by_text[adapter.serialize_full_query(turns)] = _basis(0)
    for slot, turn in enumerate(turns):
        vectors_by_text[adapter.serialize_turn_query(turn, slot=slot)] = _basis(
            slot + 1
        )
    embeddings = _embedding_rows(document_rows, turns, vectors_by_text)
    return document_rows, turns, embeddings


def test_exact_serializations_and_ties_to_even_microquantization() -> None:
    document = adapter.BlockDocument(
        unit_id="0" * 64,
        context_id="a" * 64,
        title="Raw Title",
        section_title="Raw Section",
        context_window_ordinal=7,
        text="Exact window substring.",
    )
    current = adapter.QuestionTurn("Who is current?")
    previous = adapter.QuestionTurn("Who was previous?")
    assert adapter.official_inner_unit_text(document) == (
        "TITLE:Raw Title\n"
        "SECTION:Raw Section\n"
        "TEXT:Exact window substring."
    )
    assert adapter.serialize_evidence_unit(document) == (
        '{"text":"TITLE:Raw Title\\nSECTION:Raw Section\\n'
        'TEXT:Exact window substring.",'
        f'"title":"QUAC_EVIDENCE_UNIT_{"0" * 64}"}}\n'
    )
    assert adapter.serialize_evidence_unit(document) == (
        official_contract.canonical_unit_document(
            official_contract.UnitRow(
                document.unit_id,
                adapter.official_inner_unit_text(document),
            )
        )
    )
    assert adapter.serialize_turn_query(current, slot=0) == (
        "TURN_0_CURRENT:\nWho is current?"
    )
    assert adapter.serialize_full_query((current, previous)) == (
        "TURN_0_CURRENT:\nWho is current?\n"
        "TURN_1_PREVIOUS:\nWho was previous?"
    )
    assert adapter.microquantize(0.0000005) == 0
    assert adapter.microquantize(0.0000015) == 2
    assert adapter.microquantize(-0.0000015) == -2

    # A one-turn full query is exactly its per-turn query and is encoded once.
    requests = adapter.required_embedding_serializations(
        (
            document,
            adapter.BlockDocument(
                unit_id="1" * 64,
                context_id="b" * 64,
                title="t1",
                section_title="s1",
                context_window_ordinal=0,
                text="one",
            ),
            adapter.BlockDocument(
                unit_id="2" * 64,
                context_id="c" * 64,
                title="t2",
                section_title="s2",
                context_window_ordinal=0,
                text="two",
            ),
            adapter.BlockDocument(
                unit_id="3" * 64,
                context_id="d" * 64,
                title="t3",
                section_title="s3",
                context_window_ordinal=0,
                text="three",
            ),
            adapter.BlockDocument(
                unit_id="4" * 64,
                context_id="e" * 64,
                title="t4",
                section_title="s4",
                context_window_ordinal=0,
                text="four",
            ),
        ),
        (current,),
    )
    assert len(requests) == 6
    assert len({digest for digest, _text in requests}) == len(requests)

    duplicate_text = adapter.BlockDocument(
        unit_id="1" * 64,
        context_id="f" * 64,
        title=document.title,
        section_title=document.section_title,
        context_window_ordinal=document.context_window_ordinal,
        text=document.text,
    )
    assert adapter.official_inner_unit_text(duplicate_text) == (
        adapter.official_inner_unit_text(document)
    )
    assert adapter.serialize_evidence_unit(duplicate_text) != (
        adapter.serialize_evidence_unit(document)
    )


def test_raw_direct_anchors_two_frontier_expansions_and_complete_state() -> None:
    documents, turns, embeddings = _maximal_fixture()
    result = adapter.build_action_graph(
        adapter.ActionAdapterInput(
            documents=documents,
            question_turns=turns,
            minilm_embeddings=embeddings,
        )
    )
    raw_ids = tuple(
        sorted(_opaque(f"unit:r{index}") for index in range(5))
    )
    anchor_ids = tuple(
        _opaque(f"unit:a{index}") for index in range(4)
    )
    assert result.raw_top5 == raw_ids
    assert result.direct_anchor_unit_ids == anchor_ids
    assert result.graph.unit_ids == tuple(
        sorted(
            (
                *raw_ids,
                *anchor_ids,
                _opaque("unit:x0"),
                _opaque("unit:x1"),
            )
        )
    )
    assert len(result.graph.units) == adapter.MAX_GRAPH_UNITS
    assert (
        len(set(result.graph.unit_ids).difference(result.raw_top5))
        == adapter.MAX_REPLACEMENT_CANDIDATES
    )
    assert result.complete_state_count == adapter.MAX_COMPLETE_STATES == 181
    states = evaluator.enumerate_complete_states(
        result.graph,
        raw_top5=result.raw_top5,
    )
    assert len(states) == 181
    assert sum(state.replacements == 0 for state in states) == 1
    assert sum(state.replacements == 1 for state in states) == 30
    assert sum(state.replacements == 2 for state in states) == 150

    by_id = {unit.unit_id: unit for unit in result.graph.units}
    assert by_id[anchor_ids[0]].dialogue_facets == (1, 0, 0, 0)
    assert by_id[anchor_ids[1]].dialogue_facets == (0, 1, 0, 0)
    assert by_id[anchor_ids[2]].dialogue_facets == (0, 0, 1, 0)
    assert by_id[anchor_ids[3]].dialogue_facets == (0, 0, 0, 1)
    assert by_id[anchor_ids[0]].node_features == (0.0, 1.0, 1.0, 1.0)
    assert by_id[anchor_ids[1]].node_features == (0.0, 1.0, 0.75, 1.0)
    assert by_id[_opaque("unit:x0")].node_features == (
        0.0,
        0.0,
        1.0,
        0.5,
    )
    assert by_id[_opaque("unit:x1")].node_features == (
        0.0,
        0.0,
        1.0,
        0.333333,
    )
    observed_edges = {
        frozenset((edge.left, edge.right)): (
            edge.relation,
            edge.strength,
        )
        for edge in result.graph.edges
    }
    assert observed_edges[
        frozenset((anchor_ids[0], _opaque("unit:x0")))
    ] == ("adjacent_window", 1.0)
    assert observed_edges[
        frozenset((_opaque("unit:x0"), _opaque("unit:x1")))
    ] == ("adjacent_window", 1.0)


def test_entity_parser_df_bounds_and_all_three_typed_relations() -> None:
    assert adapter.proper_name_keys(
        "Ａlice Smith met Bob. Carol, Dave and lower."
    ) == ("alice smith", "bob", "carol", "dave")
    assert adapter.proper_name_keys("A lower I abc") == ()
    assert adapter.proper_name_keys("New-York") == ("new", "york")

    documents = (
        adapter.BlockDocument(
            _opaque("unit:u0"),
            _opaque("context:c0"),
            "Exact Title",
            "Exact Section",
            0,
            "Alice met Bob and Carol with Dave.",
        ),
        adapter.BlockDocument(
            _opaque("unit:u1"),
            _opaque("context:c0"),
            "Different",
            "Different",
            1,
            "Alice met Bob and Carol with Dave.",
        ),
        adapter.BlockDocument(
            _opaque("unit:u2"),
            _opaque("context:c2"),
            "Exact Title",
            "Exact Section",
            2,
            "lowercase text only",
        ),
        adapter.BlockDocument(
            _opaque("unit:u3"),
            _opaque("context:c3"),
            "t3",
            "s3",
            0,
            "lowercase three",
        ),
        adapter.BlockDocument(
            _opaque("unit:u4"),
            _opaque("context:c4"),
            "t4",
            "s4",
            0,
            "lowercase four",
        ),
    )
    edges = adapter._frozen_typed_edges(documents)
    observed = {
        (edge.left, edge.right, edge.relation): edge.strength_micro
        for edge in edges
    }
    u0 = _opaque("unit:u0")
    u1 = _opaque("unit:u1")
    u2 = _opaque("unit:u2")
    adjacent_pair = tuple(sorted((u0, u1)))
    section_pair = tuple(sorted((u0, u2)))
    assert observed[(*adjacent_pair, "adjacent_window")] == 1_000_000
    assert observed[(*section_pair, "same_section")] == 333_333
    # Four maximal proper-name keys are shared, so the capped strength is one.
    assert observed[(*adjacent_pair, "entity_chain")] == 1_000_000

    # A key with block-window DF 17 is outside the frozen entity registry.
    too_common = tuple(
        adapter.BlockDocument(
            _opaque(f"unit:d{index:02d}"),
            _opaque(f"context:d{index:02d}"),
            f"title {index}",
            f"section {index}",
            0,
            "Alice appears here",
        )
        for index in range(17)
    )
    assert all(
        edge.relation != "entity_chain"
        for edge in adapter._frozen_typed_edges(too_common)
    )


def test_permutation_determinism_and_embedding_association_firewall() -> None:
    documents, turns, embeddings = _maximal_fixture()
    first = adapter.build_action_graph(
        adapter.ActionAdapterInput(documents, turns, embeddings)
    )
    permuted = adapter.build_action_graph(
        adapter.ActionAdapterInput(
            tuple(reversed(documents)),
            turns,
            tuple(reversed(embeddings)),
        )
    )
    assert adapter.canonical_action_bytes(first) == adapter.canonical_action_bytes(
        permuted
    )
    assert adapter.action_sha256(first) == adapter.action_sha256(permuted)

    with pytest.raises(
        adapter.QuacP1ActionAdapterError,
        match="exactly bind",
    ):
        adapter.ActionAdapterInput(documents, turns, embeddings[:-1])
    extra = adapter.MiniLmEmbedding(
        serialization_sha256=hashlib.sha256(b"not requested").hexdigest(),
        vector=_basis(8),
    )
    with pytest.raises(
        adapter.QuacP1ActionAdapterError,
        match="exactly bind",
    ):
        adapter.ActionAdapterInput(documents, turns, embeddings + (extra,))
    with pytest.raises(
        adapter.QuacP1ActionAdapterError,
        match="duplicated",
    ):
        adapter.ActionAdapterInput(
            documents,
            turns,
            embeddings + (embeddings[0],),
        )


def test_public_contract_cannot_accept_sensitive_or_pruning_parameters() -> None:
    forbidden = {
        "answer",
        "candidate",
        "family",
        "gold",
        "hipporag",
        "item_id",
        "label",
        "qrel",
        "query_context",
        "source",
        "split",
        "state",
        "utility",
    }
    for cls in (
        adapter.BlockDocument,
        adapter.QuestionTurn,
        adapter.MiniLmEmbedding,
        adapter.ActionAdapterInput,
    ):
        assert not forbidden.intersection(
            name.casefold() for name in cls.__dataclass_fields__
        )
    assert tuple(inspect.signature(adapter.build_action_graph).parameters) == (
        "action_input",
    )
    assert tuple(
        inspect.signature(adapter.required_embedding_serializations).parameters
    ) == ("documents", "question_turns")

    documents, turns, embeddings = _maximal_fixture()
    result = adapter.build_action_graph(
        adapter.ActionAdapterInput(documents, turns, embeddings)
    )
    payload = adapter.canonical_action_payload(result)
    serialized = adapter.canonical_action_bytes(result)
    assert payload["schema"] == adapter.SCHEMA
    assert b"lowercase raw evidence" not in serialized
    assert b"raw title" not in serialized
    assert b"expansion-context" not in serialized
    assert all(
        set(unit) == {
            "unit_id",
            "node_features_micro",
            "dialogue_facets",
        }
        for unit in payload["graph"]["units"]
    )
    with pytest.raises(
        adapter.QuacP1ActionAdapterError,
        match="opaque lowercase SHA-256",
    ):
        adapter.BlockDocument(
            "readable-source-coordinate",
            _opaque("valid-context"),
            "title",
            "section",
            0,
            "text",
        )
    with pytest.raises(
        adapter.QuacP1ActionAdapterError,
        match="opaque lowercase SHA-256",
    ):
        adapter.BlockDocument(
            _opaque("valid-unit"),
            "readable-source-coordinate",
            "title",
            "section",
            0,
            "text",
        )


def test_tie_breaks_use_micro_score_then_unit_id_and_recent_slot() -> None:
    documents = tuple(
        adapter.BlockDocument(
            unit_id=_opaque(f"tie-unit:{index}"),
            context_id=_opaque(f"tie-context:{index}"),
            title=f"t{index}",
            section_title=f"s{index}",
            context_window_ordinal=0,
            text=f"lowercase evidence {index}",
        )
        for index in range(6)
    )
    turns = (
        adapter.QuestionTurn("current"),
        adapter.QuestionTurn("previous"),
    )
    vectors_by_text = {
        adapter.serialize_evidence_unit(row): _basis(0)
        for row in documents
    }
    vectors_by_text[adapter.serialize_full_query(turns)] = _basis(0)
    vectors_by_text[adapter.serialize_turn_query(turns[0], slot=0)] = _basis(0)
    vectors_by_text[adapter.serialize_turn_query(turns[1], slot=1)] = _basis(0)
    embeddings = _embedding_rows(documents, turns, vectors_by_text)
    result = adapter.build_action_graph(
        adapter.ActionAdapterInput(documents, turns, embeddings)
    )
    sorted_ids = tuple(sorted(row.unit_id for row in documents))
    assert result.raw_top5 == sorted_ids[:5]
    assert result.direct_anchor_unit_ids == (sorted_ids[0], sorted_ids[0])
    by_id = {unit.unit_id: unit for unit in result.graph.units}
    assert by_id[sorted_ids[0]].dialogue_facets == (1, 1, 0, 0)
    # Equal per-turn microcosines select the more recent slot.
    assert by_id[sorted_ids[0]].node_features[2] == 1.0
