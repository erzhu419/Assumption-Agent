from pathlib import Path

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_source_qualification_v1 as source,
)


def test_frozen_sources_are_structurally_valid() -> None:
    base = Path(__file__).resolve().parents[1]
    sources = source.load_sources(base)
    assert {
        family: (len(value.document_ids), len(value.examples))
        for family, value in sources.items()
    } == {
        "EARTH_SCIENCE": (121249, 116),
        "PSYCHOLOGY": (52835, 101),
        "SUSTAINABLE_LIVING": (60792, 108),
    }
    assert all(value.filtered_document_count == 0 for value in sources.values())


def test_projection_cap_is_shared() -> None:
    base = Path(__file__).resolve().parents[1]
    sources = source.load_sources(base)
    assert all(
        len(content) <= source.DOCUMENT_CHARACTER_CAP
        for family in sources.values()
        for content in family.document_contents
    )
