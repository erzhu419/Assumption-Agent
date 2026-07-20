from __future__ import annotations

import json
from pathlib import Path

from assumption_agent.benchmarks import bright_reasoning_retrieval_acquisition_v1 as source
from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_acquisition_v1 as acquisition
from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_entrypoint_v1 as entrypoint
from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_measurement_v1 as measurement
from replication_runtime.bright_official_hipporag_v1.contract import (
    canonical_json_bytes,
    output_payload,
)


def _rows() -> tuple[source.SourceItem, ...]:
    rows = []
    for family in source.FAMILY_ORDER:
        for index in range(20):
            rows.append(
                source.SourceItem(
                    family=family,
                    source_id=f"{family}-{index}",
                    query=f"query {family} {index}",
                    excluded_ids=(),
                    gold_ids=(f"gold-{family}-{index}",),
                )
            )
    return tuple(rows)


def test_reserve_selection_is_first_15_per_family_without_gold_use() -> None:
    selected = acquisition.select_measurement_rows(_rows())
    assert len(selected) == 45
    for family_index, family in enumerate(source.FAMILY_ORDER):
        family_rows = selected[family_index * 15 : (family_index + 1) * 15]
        assert [row.source_id for row in family_rows] == [
            f"{family}-{index}" for index in range(15)
        ]


def test_view_excludes_gold_and_label_commitments_match() -> None:
    selected = acquisition.select_measurement_rows(_rows())
    view = acquisition.measurement_view(selected)
    labels = acquisition.measurement_labels(selected)
    assert "gold_ids" not in view["items"][0]
    assert labels["items"][0]["gold_ids"]
    assert [row["item_commitment_sha256"] for row in view["items"]] == [
        row["item_commitment_sha256"] for row in labels["items"]
    ]
    assert source.verify_self_hash(view, "pack_sha256") == view["pack_sha256"]
    assert source.verify_self_hash(labels, "pack_sha256") == labels["pack_sha256"]


def test_paired_counts() -> None:
    assert measurement._paired([3, 2, 1], [2, 2, 3]) == {
        "gain": 1,
        "harm": 1,
        "tie": 1,
    }


def test_entrypoint_adapters_accept_pretty_bound_json_and_logless_output(
    tmp_path: Path,
) -> None:
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps({"a": 1}, indent=2), encoding="ascii")
    assert entrypoint._read_bound_json(pretty, "pretty") == {"a": 1}
    item_root = tmp_path / "item"
    item_root.mkdir()
    payload = output_payload(
        top_ordinals=tuple(range(10)), graph_nodes=40, graph_edges=50
    )
    (item_root / "output.json").write_bytes(canonical_json_bytes(payload))
    recovered = entrypoint._recoverable_existing_hipporag(
        item_root, tuple(range(100, 132))
    )
    assert recovered["top_rows"] == list(range(100, 110))
    assert recovered["stdout_sha256"] is None
    assert recovered["stderr_sha256"] is None
