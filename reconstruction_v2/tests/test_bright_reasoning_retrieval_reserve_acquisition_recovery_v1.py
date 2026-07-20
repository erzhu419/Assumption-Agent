from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from assumption_agent.benchmarks import bright_reasoning_retrieval_acquisition_v1 as source
from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_acquisition_v1 as acquisition
from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_acquisition_recovery_v1 as recovery


def _packs() -> tuple[dict[str, object], dict[str, object]]:
    rows = []
    for family in source.FAMILY_ORDER:
        for index in range(acquisition.COUNT_PER_FAMILY):
            rows.append(
                source.SourceItem(
                    family=family,
                    source_id=f"{family}-{index}",
                    query=f"query {family} {index}",
                    excluded_ids=(f"excluded-{family}-{index}",),
                    gold_ids=(f"gold-{family}-{index}",),
                )
            )
    return acquisition.measurement_view(rows), acquisition.measurement_labels(rows)


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    body = dict(payload)
    del body["pack_sha256"]
    return source.self_hashed(body, "pack_sha256")


def test_validate_existing_packs_accepts_balanced_gold_separation() -> None:
    view, labels = _packs()
    result = recovery.validate_existing_packs(view, labels)
    assert result["family_counts"] == {
        family: acquisition.COUNT_PER_FAMILY for family in source.FAMILY_ORDER
    }
    assert result["view_pack_sha256"] == view["pack_sha256"]
    assert result["label_pack_sha256"] == labels["pack_sha256"]


def test_validate_existing_packs_rejects_commitment_mismatch() -> None:
    view, labels = _packs()
    changed = deepcopy(labels)
    changed["items"][0]["item_commitment_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(recovery.BrightReserveRecoveryError):
        recovery.validate_existing_packs(view, _rehash(changed))


def test_canonical_pack_reader_requires_acquisition_newline_contract(
    tmp_path: Path,
) -> None:
    view, _labels = _packs()
    path = tmp_path / "view.json"
    path.write_bytes(source.canonical_json_bytes(view) + b"\n")
    assert recovery._read_canonical_pack(path, "view") == view
    path.write_bytes(source.canonical_json_bytes(view))
    with pytest.raises(recovery.BrightReserveRecoveryError):
        recovery._read_canonical_pack(path, "view")
