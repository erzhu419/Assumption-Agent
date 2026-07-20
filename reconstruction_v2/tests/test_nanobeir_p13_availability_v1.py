from pathlib import Path

import numpy as np
import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_completecase_availability_v1 as mature,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p13_availability_v1 as availability,
)


def _family(count: int) -> mature.SourceFamily:
    return mature.SourceFamily(
        ids=tuple(f"d{i}" for i in range(40)),
        contents=tuple(f"document {i}" for i in range(40)),
        query_ids=tuple(f"q{i}" for i in range(count)),
        queries=tuple(f"query {i}" for i in range(count)),
    )


def test_frozen_family_counts_sum_to_149() -> None:
    assert availability.FAMILY_QUERY_COUNTS == {
        "NanoFiQA2018": 50,
        "NanoNFCorpus": 50,
        "NanoTouche2020": 49,
    }
    assert availability.ITEM_COUNT == 149


def test_build_screen_items_supports_variable_family_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        family: _family(count)
        for family, count in availability.FAMILY_QUERY_COUNTS.items()
    }
    corpus = {
        family: np.zeros((40, 384), dtype=np.float32)
        for family in availability.FAMILIES
    }
    queries = {
        family: np.zeros((count, 384), dtype=np.float32)
        for family, count in availability.FAMILY_QUERY_COUNTS.items()
    }

    class Local:
        candidate_rows = tuple(range(32))
        raw_rows = tuple(range(10))

    monkeypatch.setattr(availability.core, "build_local_retrieval", lambda _x: Local())
    items = availability.build_screen_items(sources, corpus, queries)
    assert len(items) == 149
    assert items[0].ordinal == 0
    assert items[-1].ordinal == 148
    assert items[-1].family == "NanoTouche2020"
    assert items[-1].family_ordinal == 48


def test_wrapper_context_restores_mature_screen() -> None:
    original_schema = mature.SCHEMA
    original_count = mature.ITEM_COUNT
    with availability._patched_mature_screen():
        assert mature.SCHEMA == availability.SCHEMA
        assert mature.ITEM_COUNT == 149
        assert mature.load_sources is availability.load_sources
    assert mature.SCHEMA == original_schema
    assert mature.ITEM_COUNT == original_count


def test_formal_refuses_consumed_root_before_private_access(
    tmp_path: Path,
) -> None:
    base = tmp_path / "reconstruction_v2"
    (base / availability.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(availability.OneShotRefusal, match="root already exists"):
        availability.run_formal(tmp_path)
