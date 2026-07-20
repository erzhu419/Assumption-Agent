from dataclasses import dataclass

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_bridge_expansion_core_v1 as bridge,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p13_bridge_safe_candidate_v1 as candidate,
)


@dataclass(frozen=True)
class Item:
    query: str


def _invalid_row() -> dict:
    return {
        "completion_sha256": "a" * 64,
        "completion_token_count": 10,
        "expansions": [],
        "generation_valid": False,
        "ordinal": 0,
    }


def test_safe_cap_is_exact_for_full_96_character_anchor() -> None:
    assert candidate.BRIDGE_SAFE_QUERY_CHARACTERS == 671
    assert (
        candidate.BRIDGE_SAFE_QUERY_CHARACTERS
        + 1
        + bridge.MAX_ANCHOR_CHARACTERS
        == bridge.MAX_BRIDGE_QUERY_CHARACTERS
    )


def test_role_projection_is_bounded_distinct_and_whitespace_canonical() -> None:
    source = "  alpha\n beta " + "x" * 1000
    relation = candidate.bridge_safe_query(source, "relation")
    mechanism = candidate.bridge_safe_query(source, "mechanism")
    assert relation.startswith("relation: alpha beta ")
    assert mechanism.startswith("mechanism: alpha beta ")
    assert len(relation) == len(mechanism) == 671
    assert relation != mechanism


def test_invalid_long_p12_fallback_becomes_bridge_executable() -> None:
    output = {
        "items": [_invalid_row()],
        "schema": "bright_query_generator_v1_output",
    }
    projected, audit = candidate.totalize_and_project_qwen_output(
        output, [Item("q" * 2200)]
    )
    expansions = projected["items"][0]["expansions"]
    assert len(expansions[1]) == len(expansions[2]) == 671
    assert audit["totalized_generation_count"] == 1
    assert audit["maximum_composed_bridge_query_characters"] == 768


def test_projection_retains_entity_and_constraint_bytes() -> None:
    output = {
        "items": [
            {
                "completion_sha256": "b" * 64,
                "completion_token_count": 20,
                "expansions": [
                    "entity bytes",
                    "relation bytes",
                    "mechanism bytes",
                    "constraint bytes",
                ],
                "generation_valid": True,
                "ordinal": 0,
            }
        ],
        "schema": "bright_query_generator_v1_output",
    }
    projected, _audit = candidate.totalize_and_project_qwen_output(
        output, [Item("original")]
    )
    values = projected["items"][0]["expansions"]
    assert values[0] == "entity bytes"
    assert values[3] == "constraint bytes"
    assert values[1] == "relation: relation bytes"
    assert values[2] == "mechanism: mechanism bytes"


def test_full_anchors_are_retained_and_bridge_queries_are_unique() -> None:
    anchors = tuple(
        bridge.BridgeAnchor(
            seed_row=index,
            seed_rank=index,
            sentence_rank=0,
            token_start=0,
            text=(chr(ord("a") + index) * bridge.MAX_ANCHOR_CHARACTERS),
            normalized=(chr(ord("a") + index) * bridge.MAX_ANCHOR_CHARACTERS),
        )
        for index in range(4)
    )
    queries = bridge.build_bridge_queries(
        relation_query=candidate.bridge_safe_query("r" * 1000, "relation"),
        mechanism_query=candidate.bridge_safe_query("m" * 1000, "mechanism"),
        anchors=anchors,
    )
    assert len(queries) == 4
    assert len({row.text.casefold() for row in queries}) == 4
    assert all(len(row.text) == 768 for row in queries)
    assert [row.anchor for row in queries] == [row.text[-96:] for row in queries]
