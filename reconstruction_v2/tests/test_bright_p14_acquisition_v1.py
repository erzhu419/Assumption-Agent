from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_acquisition_v1 as acquisition,
)


def test_allocation_uses_frozen_72_position_windows() -> None:
    blocks = acquisition.allocate_windows(
        bytes(range(32)),
        acquisition.FAMILIES[0],
        tuple(f"q{i}" for i in range(100)),
    )
    assert {name: len(values) for name, values in blocks.items()} == {
        "C_confirm": 20,
        "A_form": 16,
        "F_search": 10,
        "A_hold": 10,
        "M_search": 10,
        "RESERVE": 6,
    }
    flattened = tuple(value for rows in blocks.values() for value in rows)
    assert flattened == acquisition.hmac_order(
        bytes(range(32)),
        acquisition.FAMILIES[0],
        tuple(f"q{i}" for i in range(100)),
    )[:72]


def test_item_commitment_is_deterministic_and_family_separated() -> None:
    secret = bytes(range(32))
    assert acquisition._item_key(secret, "EARTH_SCIENCE", "q1") == (
        acquisition._item_key(secret, "EARTH_SCIENCE", "q1")
    )
    assert acquisition._item_key(secret, "EARTH_SCIENCE", "q1") != (
        acquisition._item_key(secret, "PSYCHOLOGY", "q1")
    )


def test_formal_refuses_consumed_root_before_private_access(
    tmp_path: Path,
) -> None:
    base = tmp_path / "reconstruction_v2"
    (base / acquisition.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(acquisition.OneShotRefusal, match="root already exists"):
        acquisition.run_formal(tmp_path)
