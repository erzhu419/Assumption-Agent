from __future__ import annotations

from collections import Counter

import pytest

from assumption_agent.benchmarks import bright_reasoning_retrieval_acquisition_v1 as acquisition


def _rows(family: str, count: int = 103):
    return [
        {
            "query": f"{family} private query {index}",
            "id": f"{family.lower()}-{index:03d}",
            "excluded_ids": [f"excluded-{index}"],
            "gold_ids": [f"gold-{index}", f"gold-{index}-b"],
        }
        for index in range(count)
    ]


def _items():
    return {
        family: acquisition.decode_source_rows(family, _rows(family))
        for family in acquisition.FAMILY_ORDER
    }


def test_hmac_assignment_is_complete_balanced_disjoint_and_deterministic() -> None:
    items = _items()
    first = acquisition.assign_blocks(items, b"a" * 32)
    second = acquisition.assign_blocks(items, b"a" * 32)
    different = acquisition.assign_blocks(items, b"b" * 32)
    assert first == second
    assert any(
        [row.commitment_sha256 for row in first[block]]
        != [row.commitment_sha256 for row in different[block]]
        for block in acquisition.BLOCK_ORDER
    )
    flattened = [row.commitment_sha256 for rows in first.values() for row in rows]
    assert len(flattened) == len(set(flattened)) == 309
    for block in acquisition.BLOCK_ORDER:
        assert Counter(row.family for row in first[block]) == Counter(
            {family: acquisition.BLOCK_COUNTS[block] for family in acquisition.FAMILY_ORDER}
        )
    assert len(first["RESERVE"]) == 3 * (103 - 75)


def test_view_and_label_packs_are_joinable_but_field_separated() -> None:
    rows = acquisition.assign_blocks(_items(), b"c" * 32)["M_search"]
    view = acquisition.block_view("M_search", rows)
    labels = acquisition.block_labels("M_search", rows)
    assert acquisition.verify_self_hash(view, "pack_sha256")
    assert acquisition.verify_self_hash(labels, "pack_sha256")
    assert view["item_count"] == labels["item_count"] == 45
    assert [row["item_commitment_sha256"] for row in view["items"]] == [
        row["item_commitment_sha256"] for row in labels["items"]
    ]
    assert all(set(row) == {"excluded_ids", "family", "item_commitment_sha256", "ordinal", "query"} for row in view["items"])
    assert all(set(row) == {"gold_ids", "item_commitment_sha256", "ordinal"} for row in labels["items"])
    serialized_items = acquisition.canonical_json_bytes(view["items"])
    assert b"gold_ids" not in serialized_items
    assert b"source_id" not in serialized_items


def test_reserve_has_no_label_pack_and_source_identity_is_not_exported() -> None:
    rows = acquisition.assign_blocks(_items(), b"d" * 32)["RESERVE"]
    view = acquisition.block_view("RESERVE", rows)
    assert view["item_count"] == 84
    assert all("source_id" not in row and "gold_ids" not in row for row in view["items"])
    with pytest.raises(acquisition.BrightAcquisitionError, match="label block"):
        acquisition.block_labels("RESERVE", rows)


def test_duplicate_equivalence_keys_fail_closed() -> None:
    rows = _rows("BIOLOGY")
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(acquisition.BrightAcquisitionError, match="source ids"):
        acquisition.decode_source_rows("BIOLOGY", rows)
    rows = _rows("BIOLOGY")
    rows[1]["query"] = "  BIOLOGY PRIVATE QUERY 0  "
    with pytest.raises(acquisition.BrightAcquisitionError, match="normalized queries"):
        acquisition.decode_source_rows("BIOLOGY", rows)
    rows = _rows("BIOLOGY")
    rows[1]["gold_ids"] = list(reversed(rows[0]["gold_ids"]))
    with pytest.raises(acquisition.BrightAcquisitionError, match="gold id sets"):
        acquisition.decode_source_rows("BIOLOGY", rows)


def test_secret_and_projected_row_contracts_fail_closed() -> None:
    one = acquisition.decode_source_rows("BIOLOGY", _rows("BIOLOGY", 1))[0]
    with pytest.raises(acquisition.BrightAcquisitionError, match="32 bytes"):
        acquisition.selection_priority(b"short", one)
    malformed = _rows("BIOLOGY", 1)
    malformed[0]["reasoning"] = "forbidden"
    with pytest.raises(acquisition.BrightAcquisitionError, match="shape"):
        acquisition.decode_source_rows("BIOLOGY", malformed)
