from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_completecase_availability_v1 as availability,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "reconstruction_v2"


def test_frozen_preconditions_and_source_file_scope() -> None:
    loaded = availability._verify_preconditions(BASE)
    assert loaded["source_access"]["qualification"]["source_passed"] is True
    assert set(availability.SOURCE_FILES) == {
        f"{role}/{family}-00000-of-00001.parquet"
        for role in ("corpus", "queries")
        for family in availability.FAMILIES
    }
    source = inspect.getsource(availability.load_sources)
    assert '"qrels"' not in source
    assert "qrel_table" not in source


def test_load_sources_never_reads_qrel_member(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[Path] = []
    original = pq.read_table

    def read_table(path: object, *args: object, **kwargs: object):
        observed.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", read_table)
    loaded = availability.load_sources(BASE)
    assert tuple(loaded) == availability.FAMILIES
    assert len(observed) == 6
    assert all(path.parent.name in {"corpus", "queries"} for path in observed)
    assert all(len(loaded[family].query_ids) == 50 for family in availability.FAMILIES)


def _synthetic_sources() -> dict[str, availability.SourceFamily]:
    result: dict[str, availability.SourceFamily] = {}
    for family in availability.FAMILIES:
        result[family] = availability.SourceFamily(
            ids=tuple(f"{family}-doc-{index}" for index in range(64)),
            contents=tuple(f"document {index}" for index in range(64)),
            query_ids=tuple(f"{family}-query-{index}" for index in range(50)),
            queries=tuple(f"query {index}" for index in range(50)),
        )
    return result


def test_build_screen_items_is_deterministic_and_complete() -> None:
    sources = _synthetic_sources()
    corpus_embeddings = {
        family: np.zeros((64, 384), dtype=np.float32)
        for family in availability.FAMILIES
    }
    query_embeddings = {
        family: np.zeros((50, 384), dtype=np.float32)
        for family in availability.FAMILIES
    }
    first = availability.build_screen_items(
        sources, corpus_embeddings, query_embeddings
    )
    second = availability.build_screen_items(
        sources, corpus_embeddings, query_embeddings
    )
    assert first == second
    assert len(first) == availability.ITEM_COUNT
    assert [item.ordinal for item in first] == list(range(150))
    assert all(item.base_pool == tuple(range(32)) for item in first)
    assert all(item.raw_top10 == tuple(range(10)) for item in first)


def test_prepare_item_roots_creates_all_parents_before_execution(
    tmp_path: Path,
) -> None:
    sources = _synthetic_sources()
    items = tuple(
        availability.ScreenItem(
            ordinal=index,
            family=availability.FAMILIES[0],
            family_ordinal=index,
            query_id=f"query-{index}",
            query=f"query {index}",
            base_pool=tuple(range(32)),
            raw_top10=tuple(range(10)),
        )
        for index in range(3)
    )
    roots = availability._prepare_item_roots(
        root=tmp_path, items=items, sources=sources
    )
    assert set(roots) == {0, 1, 2}
    for root in roots.values():
        assert (root / "home").is_dir()
        assert (root / "hf").is_dir()
        assert (root / "tmp").is_dir()
        assert (root / "input.json").is_file()


def test_failure_row_is_label_free(tmp_path: Path) -> None:
    item = availability.ScreenItem(
        ordinal=0,
        family=availability.FAMILIES[0],
        family_ordinal=0,
        query_id="query-id",
        query="query text",
        base_pool=tuple(range(32)),
        raw_top10=tuple(range(10)),
    )
    row = availability._failure_row(item, tmp_path, RuntimeError("fixture"))
    assert row["availability"] == "failed"
    assert row["exception_class"] == "RuntimeError"
    assert not ({"label", "qrel", "score", "gold"} & set(row))


def test_one_shot_refusal_precedes_freeze_read(tmp_path: Path) -> None:
    project = tmp_path / "project"
    base = project / "reconstruction_v2"
    (base / availability.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(availability.OneShotRefusal):
        availability.run_formal(project)
