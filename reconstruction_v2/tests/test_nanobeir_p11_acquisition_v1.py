from __future__ import annotations

from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p11_acquisition_v1 as acquisition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "reconstruction_v2"


def test_hmac_allocation_is_deterministic_disjoint_and_complete() -> None:
    secret = bytes(range(32))
    query_ids = [f"q{index:02d}" for index in range(50)]
    first = acquisition.allocate_blocks(secret, acquisition.FAMILIES[0], query_ids)
    second = acquisition.allocate_blocks(secret, acquisition.FAMILIES[0], query_ids)
    assert first == second
    assert {name: len(rows) for name, rows in first.items()} == {
        "C_confirm": 12,
        "A_form": 10,
        "F_search": 8,
        "A_hold": 8,
        "M_search": 8,
        "RESERVE": 4,
    }
    flattened = [item for rows in first.values() for item in rows]
    assert len(flattened) == len(set(flattened)) == 50


def test_hmac_family_domain_separation() -> None:
    secret = bytes(range(32))
    query_ids = [f"q{index:02d}" for index in range(50)]
    left = acquisition.hmac_order(secret, acquisition.FAMILIES[0], query_ids)
    right = acquisition.hmac_order(secret, acquisition.FAMILIES[1], query_ids)
    assert left != right


def test_allocation_rejects_smaller_or_larger_family() -> None:
    secret = bytes(range(32))
    for count in (49, 51):
        with pytest.raises(acquisition.NanoBEIRAcquisitionError):
            acquisition.allocate_blocks(
                secret,
                acquisition.FAMILIES[0],
                [f"q{index}" for index in range(count)],
            )


def test_shared_document_projection() -> None:
    assert acquisition.project_document("a" * 4000) == "a" * 3000
    assert acquisition.project_document("short") == "short"
    with pytest.raises(acquisition.NanoBEIRAcquisitionError):
        acquisition.project_document("   ")


def test_real_source_families_have_exact_capacity() -> None:
    acquisition._verify_preconditions(BASE)
    for family in acquisition.FAMILIES:
        queries, qrels, projected = acquisition._read_family(BASE, family)
        assert len(queries) == len(qrels) == 50
        assert projected >= 0
        assert all(qrels[row["query_id"]] for row in queries)


def test_pack_self_hash_round_trip() -> None:
    value = acquisition.self_hashed({"schema": "fixture"}, field="pack_sha256")
    body = dict(value)
    declared = body.pop("pack_sha256")
    assert declared == acquisition.stable_hash(body)


def test_formal_is_one_shot_before_precondition_access(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root = project / "reconstruction_v2" / acquisition.RUN_ROOT_RELATIVE
    root.mkdir(parents=True)
    with pytest.raises(acquisition.OneShotRefusal):
        acquisition.run_formal(project)
