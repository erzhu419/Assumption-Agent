import hashlib
import hmac
from pathlib import Path
import sys
import types

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p15_extension_acquisition_v1 as acquisition,
)


def test_extension_is_exact_contiguous_hmac_slice() -> None:
    secret = bytes(range(32))
    query_ids = tuple(f"q-{index:03d}" for index in range(116))
    ordered = tuple(
        sorted(
            query_ids,
            key=lambda query_id: (
                hmac.new(
                    secret,
                    (
                        acquisition.FAMILIES[0] + "\n" + query_id
                    ).encode(),
                    hashlib.sha256,
                ).digest(),
                query_id,
            ),
        )
    )
    selected = acquisition.select_extension(
        secret, acquisition.FAMILIES[0], query_ids
    )
    assert selected == ordered[72:92]
    assert not set(selected).intersection(ordered[:72])


def test_design_self_uses_newline_inclusive_canonical_sha256() -> None:
    body = {"schema": "test_design"}
    expected = hashlib.sha256(
        acquisition.p14_acquisition.utilities.canonical_json_bytes(body)
    ).hexdigest()
    acquisition._verify_design_self(
        {**body, "self_sha256": expected}, expected, "test design"
    )
    with pytest.raises(acquisition.P15AcquisitionError, match="self hash drifted"):
        acquisition._verify_design_self(
            {**body, "self_sha256": "0" * 64}, expected, "test design"
        )


def test_view_source_loader_never_requests_gold_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path
    calls: list[tuple[str, ...]] = []

    class FakeTable:
        column_names = ["query", "id", "excluded_ids"]

        def to_pylist(self):
            return [
                {
                    "query": f"query {index}",
                    "id": f"id-{index}",
                    "excluded_ids": [],
                }
                for index in range(92)
            ]

    def read_table(_path: Path, *, columns):
        calls.append(tuple(columns))
        return FakeTable()

    pyarrow = types.ModuleType("pyarrow")
    parquet = types.ModuleType("pyarrow.parquet")
    parquet.read_table = read_table
    pyarrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    monkeypatch.setattr(
        acquisition.p14_acquisition.utilities,
        "file_sha256",
        lambda path: acquisition.source.SOURCE_FILES[
            f"examples/{path.name}"
        ]["sha256"],
    )
    for family in acquisition.FAMILIES:
        slug = acquisition.source.SLUGS[family]
        path = (
            base
            / acquisition.source.SOURCE_ROOT_RELATIVE
            / "examples"
            / f"{slug}-00000-of-00001.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        size = acquisition.source.SOURCE_FILES[
            f"examples/{slug}-00000-of-00001.parquet"
        ]["size_bytes"]
        path.write_bytes(b"0" * size)
    loaded = acquisition.load_view_sources(base)
    assert all(len(rows) == 92 for rows in loaded.values())
    assert calls == [("query", "id", "excluded_ids")] * 3


def test_load_views_requires_positions_72_through_91(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path
    private = base / "private"
    private.mkdir()
    rows = []
    for family in acquisition.FAMILIES:
        for attempt in range(acquisition.ATTEMPTS_PER_FAMILY):
            rows.append(
                {
                    "attempt_ordinal": attempt,
                    "excluded_document_ids": [],
                    "family": family,
                    "family_HMAC_position": 72 + attempt,
                    "item_key": f"{family}-{attempt}",
                    "query": "query",
                    "source_query_id": f"{family}-q-{attempt}",
                }
            )
    binding = acquisition._write_view_pack(base, private, rows)
    result = {"pack_bindings": {"C_confirm_view": binding}}
    items = acquisition.load_views(base, result)
    assert len(items) == 60
    assert items[0].family_hmac_position == 72
    assert items[-1].family_hmac_position == 91


def test_consumed_root_refuses_before_private_access(tmp_path: Path) -> None:
    base = tmp_path / "reconstruction_v2"
    (base / acquisition.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(acquisition.OneShotRefusal, match="root already exists"):
        acquisition.run_formal(tmp_path)
