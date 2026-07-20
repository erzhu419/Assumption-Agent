from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_completecase_acquisition_v1 as acquisition,
)


def test_hmac_allocation_is_deterministic_disjoint_and_exact() -> None:
    secret = bytes(range(32))
    query_ids = tuple(f"q{index:02d}" for index in range(50))
    first = acquisition.allocate_blocks(secret, acquisition.FAMILIES[0], query_ids)
    second = acquisition.allocate_blocks(secret, acquisition.FAMILIES[0], query_ids)
    assert first == second
    assert {name: len(rows) for name, rows in first.items()} == {
        "C_confirm": 10,
        "A_form": 8,
        "F_search": 5,
        "A_hold": 5,
        "M_search": 5,
        "RESERVE": 3,
    }
    flattened = [value for rows in first.values() for value in rows]
    assert len(flattened) == 36
    assert len(set(flattened)) == 36


def test_hmac_allocation_selects_only_first_36_from_frozen_order() -> None:
    secret = b"z" * 32
    query_ids = tuple(f"q{index:02d}" for index in range(47))
    ordered = acquisition.hmac_order(secret, acquisition.FAMILIES[1], query_ids)
    blocks = acquisition.allocate_blocks(secret, acquisition.FAMILIES[1], query_ids)
    flattened = tuple(value for rows in blocks.values() for value in rows)
    assert flattened == ordered[:36]


@pytest.mark.parametrize("count", [0, 35])
def test_allocation_rejects_insufficient_eligible_capacity(count: int) -> None:
    with pytest.raises(
        acquisition.CompleteCaseAcquisitionError,
        match="capacity is below 36",
    ):
        acquisition.allocate_blocks(
            b"x" * 32,
            acquisition.FAMILIES[0],
            tuple(f"q{index}" for index in range(count)),
        )


def test_hmac_order_rejects_duplicate_query_ids() -> None:
    with pytest.raises(
        acquisition.CompleteCaseAcquisitionError,
        match="duplicated",
    ):
        acquisition.hmac_order(
            b"x" * 32, acquisition.FAMILIES[0], ("same", "same")
        )


def test_formal_refuses_consumed_root_before_access(tmp_path: Path) -> None:
    base = tmp_path / "reconstruction_v2"
    (base / acquisition.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(acquisition.OneShotRefusal, match="root already exists"):
        acquisition.run_formal(tmp_path)


def test_block_contract_totals_exactly_36() -> None:
    assert sum(count for _name, count in acquisition.BLOCK_COUNTS) == 33
    assert (
        sum(count for _name, count in acquisition.BLOCK_COUNTS)
        + acquisition.RESERVE_COUNT
        == acquisition.SELECTED_PER_FAMILY
    )
