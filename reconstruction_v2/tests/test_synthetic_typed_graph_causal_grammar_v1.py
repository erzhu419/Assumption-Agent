from __future__ import annotations

import ast
from collections import Counter
import hashlib
import hmac
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import contractnli_typed_clause_graph_v1 as core
from assumption_agent.benchmarks import synthetic_typed_graph_causal_grammar_v1 as g


# This is a public unit-test fixture.  It is not a formal study seed and must
# never be copied into a seed-custody or acquisition artifact.
TEST_SEED = bytes(range(32))

EXPECTED_BLOCK_COMMITMENTS = {
    "A_form": "6f54b12d6944f700635e6132d821c9011e562c32f0abb6684917188183233364",
    "F_search": "4b96c888dd07719a25d3d7aaff143d51ff706c0c33df383473bab8bf138b7326",
    "A_hold": "9f89760de2cb52af9f62a961d57ba5f89ece00de06f64df529504a132ee7cd95",
    "M_search": "5f47e64c1b8ff2d75ba9b39d44892cfd974991ca4856efc0e72c4e5608af43f5",
}


def _core_edges(item: g.CompiledItem) -> tuple[g.SyntheticEdge, ...]:
    spans = tuple(
        core.SourceSpan(node.span_i, node.start, node.end, node.identity_text)
        for node in item.nodes
    )
    return tuple(
        g.SyntheticEdge(edge.edge_family, edge.left_span_i, edge.right_span_i)
        for edge in core.build_typed_clause_graph(spans)
    )


def _degree_multiset(edges: tuple[g.SyntheticEdge, ...]) -> list[int]:
    degrees: Counter[int] = Counter()
    for edge in edges:
        degrees[edge.left_span_i] += 1
        degrees[edge.right_span_i] += 1
    return sorted(degrees.values())


def test_registry_is_exactly_four_edges_times_six_frozen_roles() -> None:
    assert len(g.FAMILY_REGISTRY) == 24
    assert len(g.FAMILY_BY_ID) == 24
    expected_roles = {
        g.TRAIN_POSITIVE_1,
        g.TRAIN_POSITIVE_2,
        g.TRAIN_NEGATIVE_1,
        g.TRAIN_NEGATIVE_2,
        g.FAMILYOUT_POSITIVE,
        g.FAMILYOUT_NEGATIVE,
    }
    for edge_family in g.EDGE_FAMILIES:
        rows = [row for row in g.FAMILY_REGISTRY if row.edge_family == edge_family]
        assert len(rows) == 6
        assert {row.family_role for row in rows} == expected_roles
        assert Counter(row.polarity for row in rows) == {g.POSITIVE: 3, g.NEGATIVE: 3}
        assert Counter(row.template_split for row in rows) == {
            g.TRAIN_SPLIT: 4,
            g.FAMILYOUT_SPLIT: 2,
        }
        for row in rows:
            mate = g.FAMILY_BY_ID[row.matched_family_id]
            assert mate.matched_family_id == row.family_id
            assert mate.edge_family == row.edge_family
            assert mate.surface_variant == row.surface_variant
            assert mate.match_group == row.match_group
            assert mate.polarity != row.polarity
            assert (row.negative_kind is None) == (row.polarity == g.POSITIVE)
            if row.family_role == g.TRAIN_NEGATIVE_2:
                assert row.negative_kind == (
                    "edge_present_but_query_and_gold_are_independent_direct_cue"
                )
                assert "decoy" not in row.negative_kind


def test_quotas_are_16_train_times_4_and_8_familyout_times_8() -> None:
    train_sets: list[set[str]] = []
    for block in ("A_form", "F_search", "A_hold"):
        quota = g.family_quota(block)
        assert len(quota) == 16
        assert {count for _family, count in quota} == {4}
        assert sum(count for _family, count in quota) == 64
        train_sets.append({family for family, _count in quota})
        assert all(
            g.FAMILY_BY_ID[family].template_split == g.TRAIN_SPLIT
            for family, _count in quota
        )
    assert train_sets[0] == train_sets[1] == train_sets[2]

    familyout = g.family_quota("M_search")
    assert len(familyout) == 8
    assert {count for _family, count in familyout} == {8}
    assert sum(count for _family, count in familyout) == 64
    familyout_set = {family for family, _count in familyout}
    assert familyout_set.isdisjoint(train_sets[0])
    assert all(
        g.FAMILY_BY_ID[family].template_split == g.FAMILYOUT_SPLIT
        for family in familyout_set
    )


def test_public_hmac_field_draw_is_exact_and_domain_separated() -> None:
    observed = g.field_digest(
        TEST_SEED,
        block="A_form",
        family_key="DEF_V1",
        slot=2,
        field="node_order",
        counter=7,
    )
    message = json.dumps(
        [g.DOMAIN, "A_form", "DEF_V1", 2, "node_order", 7],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert observed == hmac.new(TEST_SEED, message, hashlib.sha256).digest()
    assert observed != g.field_digest(
        TEST_SEED,
        block="F_search",
        family_key="DEF_V1",
        slot=2,
        field="node_order",
        counter=7,
    )
    with pytest.raises(g.SyntheticCausalGrammarError, match="32 raw bytes"):
        g.field_digest(
            b"short",
            block="A_form",
            family_key="DEF_V1",
            slot=2,
            field="node_order",
        )


def test_public_lexicon_is_fixed_width_and_injective() -> None:
    lexemes = [g.public_lexeme(index) for index in range(4096)]
    assert len(lexemes) == len(set(lexemes))
    assert len({len(value) for value in lexemes}) == 1
    assert all(value.replace("-", "").isalpha() and value.isascii() for value in lexemes)


def test_direct_generation_is_deterministic_and_matches_public_golden_hash() -> None:
    first = g.generate_all_blocks(TEST_SEED)
    second = g.generate_all_blocks(TEST_SEED)
    assert first == second
    assert {
        block: g.block_commitment(first[block]) for block in g.BLOCK_ORDER
    } == EXPECTED_BLOCK_COMMITMENTS
    assert g.generate_all_blocks(bytes(reversed(TEST_SEED))) != first


def test_all_256_slots_are_disjoint_and_have_exact_gold_and_offsets() -> None:
    blocks = g.generate_all_blocks(TEST_SEED)
    items = [item for block in g.BLOCK_ORDER for item in blocks[block]]
    assert len(items) == 256
    assert len({item.item_commitment_sha256 for item in items}) == 256
    assert len({item.label_free_commitment_sha256 for item in items}) == 256
    for block in g.BLOCK_ORDER:
        assert [item.block_ordinal for item in blocks[block]] == list(range(64))
    for item in items:
        assert len(item.nodes) == 32
        assert 1 <= len(item.gold_node_indices) <= 3
        assert tuple(sorted(set(item.gold_node_indices))) == item.gold_node_indices
        assert [node.span_i for node in item.nodes] == list(range(32))
        assert all(
            item.context[node.start : node.end] == node.identity_text
            for node in item.nodes
        )
        if item.polarity == g.POSITIVE:
            assert all(
                item.nodes[index].latent_role.startswith("causal_target_")
                for index in item.gold_node_indices
            )
        else:
            assert all(
                item.nodes[index].latent_role.startswith("direct_gold_")
                for index in item.gold_node_indices
            )


def test_every_family_compiles_the_designated_edge_in_existing_graph_core() -> None:
    for block in ("A_form", "M_search"):
        items = g.generate_block(TEST_SEED, block)
        representative_by_family: dict[str, g.CompiledItem] = {}
        for item in items:
            representative_by_family.setdefault(item.family_id, item)
        assert len(representative_by_family) == (16 if block == "A_form" else 8)
        for item in representative_by_family.values():
            graph_edges = set(_core_edges(item))
            assert set(item.designated_edges) <= graph_edges
            assert all(edge.edge_family == item.edge_family for edge in item.designated_edges)


def test_positive_negative_pairs_have_identical_structural_draws_not_content() -> None:
    for block in g.BLOCK_ORDER:
        items = g.generate_block(TEST_SEED, block)
        by_coordinate = {
            (item.family_id, item.family_slot): item for item in items
        }
        for item in items:
            mate_id = g.FAMILY_BY_ID[item.family_id].matched_family_id
            mate = by_coordinate[(mate_id, item.family_slot)]
            assert item.pair_key == mate.pair_key
            assert item.matching_signature_sha256 == mate.matching_signature_sha256
            assert item.structural_draw_sha256 == mate.structural_draw_sha256
            assert len(item.nodes) == len(mate.nodes) == 32
            assert len(item.gold_node_indices) == len(mate.gold_node_indices)
            assert item.edge_family == mate.edge_family
            assert item.item_commitment_sha256 != mate.item_commitment_sha256
            assert item.question != mate.question
            assert item.polarity != mate.polarity


def test_graph_ablations_change_only_edges_and_obey_frozen_interventions() -> None:
    items = g.generate_block(TEST_SEED, "A_form")
    for item in items:
        original_content = (
            item.question,
            item.context,
            item.nodes,
            item.gold_node_indices,
            item.label_free_commitment_sha256,
        )
        full = g.apply_graph_ablation(item, _core_edges(item), mode=g.FULL_GRAPH)
        dropped = g.apply_graph_ablation(
            item, full, mode=g.DROP_DESIGNATED
        )
        wrong = g.apply_graph_ablation(item, full, mode=g.WRONG_TYPE)
        permuted = g.apply_graph_ablation(
            item, full, mode=g.ENDPOINT_PERMUTED
        )
        assert set(item.designated_edges) <= set(full)
        assert set(item.designated_edges).isdisjoint(dropped)
        assert len(full) - len(dropped) == len(set(item.designated_edges))
        assert len(wrong) == len(full)
        assert not set(item.designated_edges) <= set(wrong)
        assert len(permuted) == len(full)
        assert _degree_multiset(permuted) == _degree_multiset(full)
        assert permuted != full
        assert original_content == (
            item.question,
            item.context,
            item.nodes,
            item.gold_node_indices,
            item.label_free_commitment_sha256,
        )


def test_unique_evaluator_derangement_is_deterministic_fixed_point_free_and_stratified() -> None:
    a_form = g.generate_block(TEST_SEED, "A_form")
    first = g.evaluator_label_derangement(a_form, seed=TEST_SEED)
    second = g.evaluator_label_derangement(a_form, seed=TEST_SEED)
    assert first == second
    assert len(first) == 64
    assert len({destination for destination, _source in first}) == 64
    assert len({source for _destination, source in first}) == 64
    by_commitment = {item.label_free_commitment_sha256: item for item in a_form}
    for destination, source in first:
        assert destination != source
        destination_item = by_commitment[destination]
        source_item = by_commitment[source]
        assert (
            destination_item.edge_family,
            len(destination_item.nodes),
            len(destination_item.gold_node_indices),
        ) == (
            source_item.edge_family,
            len(source_item.nodes),
            len(source_item.gold_node_indices),
        )
    assert g.E00_CONTROL_EVALUATOR_ID == "E_UNIFORM_L025"


def test_derangement_identifiers_and_order_are_label_free() -> None:
    source = Path(g.__file__).read_text(encoding="utf-8")
    function = source[source.index("def evaluator_label_derangement(") :]
    function = function[: function.index("\ndef block_commitment(")]
    assert "label_free_commitment_sha256" in function
    assert "item_commitment_sha256" not in function


def test_grammar_source_has_only_frozen_stdlib_imports_and_no_source_loader() -> None:
    path = Path(g.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module is not None
            imports.add(node.module.split(".", 1)[0])
    assert imports == {"dataclasses", "hashlib", "hmac", "json", "typing", "__future__"}
    assert "import random" not in source
    assert "import os" not in source
    assert "subprocess" not in source
    assert "ZipFile" not in source
    assert "open(" not in source
    assert "candidate_pool" not in source


def test_invalid_coordinates_fail_instead_of_filtering_or_replacing() -> None:
    with pytest.raises(g.SyntheticCausalGrammarError, match="family slot"):
        g.build_latent_world(
            seed=TEST_SEED,
            block="A_form",
            block_ordinal=0,
            family_id="DEF_TP1",
            family_slot=4,
        )
    with pytest.raises(g.SyntheticCausalGrammarError, match="block"):
        g.generate_block(TEST_SEED, "backup")
    item = g.generate_block(TEST_SEED, "A_form")[0]
    with pytest.raises(g.SyntheticCausalGrammarError, match="mode"):
        g.apply_graph_ablation(item, _core_edges(item), mode="adaptive")
